"""电商合规审查路由（FastAPI 应用入口，默认端口 8002）。

与 query_router.py 保持一致的工程约定，提供面向电商合规审查的一组接口：

- ``POST /review``                       提交审查请求（商家名、审查类型、问题描述）
- ``GET  /review/{review_id}``           获取审查结果（含证据链、风险等级、报告）
- ``POST /review/{review_id}/approve``   人工审批（HITL）通过 / 驳回
- ``GET  /graph/entity/{entity_id}``     查询实体画像及其关系
- ``GET  /graph/path``                   查询两个实体之间的多跳路径
- ``GET  /stream/{task_id}``             SSE 流式推送审查进度
- ``GET  /status/{task_id}``             轮询任务状态（含已完成节点中文名）

设计要点：
1. 非流式审查放到线程池执行（run_in_threadpool），避免阻塞事件循环；
2. HITL 触发时图会被 interrupt 暂停，此时 ``POST /review`` 正常返回、任务状态为
   ``pending_approval``，前端据此后调 ``/review/{id}/approve`` 恢复执行；
3. 图谱查询在无 Neo4j 驱动时返回空结果而非报错（离线友好）。
"""

import os
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from knowledge.utils.auth_util import install_security
from knowledge.processor.review_process.core.paths import get_front_page_dir
from knowledge.processor.review_process.core.deps import get_review_service
from knowledge.processor.review_process.schema.review_schema import (
    ReviewSubmitRequest, ReviewSubmitResponse, ReviewResultResponse,
    ApproveRequest, GraphEntityResponse, GraphPathResponse, ReviewTaskStatusResponse,
    TraceResponse, MetricsResponse,
)
from knowledge.processor.review_process.services.review_service import ReviewService
from knowledge.utils.sse_util import sse_generator
from knowledge.utils.observability import snapshot_metrics
from knowledge.processor.review_process.base import setup_logging


def create_app() -> FastAPI:
    app = FastAPI(title="Compliance Review Service", description="电商合规审查服务")
    install_security(app)
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))
    register_routes(app)
    return app


def register_routes(app: FastAPI):

    @app.get("/")
    async def index():
        """根路径直接打开审查控制台，省去手输 /review.html。"""
        return FileResponse(os.path.join(get_front_page_dir(), "review.html"))

    @app.get("/review.html")
    async def review_page():
        return FileResponse(os.path.join(get_front_page_dir(), "review.html"))

    # ==================== 提交审查 ====================
    @app.post("/review")
    async def submit_review(
        request: ReviewSubmitRequest,
        background_tasks: BackgroundTasks,
        service: ReviewService = Depends(get_review_service),
    ):
        """提交电商合规审查请求。

        - ``is_stream=True``：立即返回 {review_id, task_id}，审查在后台执行，
          进度通过 ``GET /stream/{task_id}`` 推送；
        - ``is_stream=False``：同步等待审查完成（放线程池执行），直接返回完整结果。
        """
        review_id = service.generate_review_id()
        task_id = service.generate_task_id()
        session_id = request.session_id or service.generate_session_id()
        service.prepare(task_id, request.is_stream)

        def _run():
            service.run_review_graph(
                review_id=review_id, task_id=task_id,
                company=request.company, review_type=request.review_type,
                question=request.question, is_stream=request.is_stream,
                session_id=session_id,
                metrics=request.metrics, relations=request.relations,
            )

        if request.is_stream:
            background_tasks.add_task(_run)
            return ReviewSubmitResponse(
                message="审查已提交", review_id=review_id, task_id=task_id,
                session_id=session_id, status="processing",
            )

        # 同步阻塞的图执行放到线程池，别卡住事件循环
        await run_in_threadpool(_run)
        result = service.get_review(review_id)
        result["session_id"] = session_id
        return ReviewResultResponse(**{
            k: v for k, v in result.items()
            if k in ReviewResultResponse.model_fields
        })

    # ==================== 获取审查结果 ====================
    @app.get("/review/{review_id}", response_model=ReviewResultResponse)
    async def get_review(
        review_id: str,
        service: ReviewService = Depends(get_review_service),
    ) -> ReviewResultResponse:
        result = await run_in_threadpool(service.get_review, review_id)
        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=f"审查记录不存在: {review_id}")
        return ReviewResultResponse(**{
            k: v for k, v in result.items()
            if k in ReviewResultResponse.model_fields
        })

    # ==================== 人工审批（HITL） ====================
    @app.post("/review/{review_id}/approve", response_model=ReviewResultResponse)
    async def approve_review(
        review_id: str,
        body: ApproveRequest,
        service: ReviewService = Depends(get_review_service),
    ) -> ReviewResultResponse:
        """人工审批：恢复被 interrupt 暂停的审查图继续执行并生成报告。"""
        result = await run_in_threadpool(
            service.approve, review_id, body.approved, body.reviewer, body.comment
        )
        if result.get("status") == "failed" and result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return ReviewResultResponse(**{
            k: v for k, v in result.items()
            if k in ReviewResultResponse.model_fields
        })

    # ==================== 可观测性：链路追踪 / 指标 ====================
    @app.get("/trace/{trace_id}", response_model=TraceResponse)
    async def get_trace(
        trace_id: str,
        service: ReviewService = Depends(get_review_service),
    ) -> TraceResponse:
        """查询一次审查的节点耗时瀑布。

        线上排障用法：``POST /review`` 返回体里的 ``trace_id`` 直接拼到本接口即可，
        一眼看出慢在检索、图谱还是 LLM。
        """
        result = await run_in_threadpool(service.get_trace, trace_id)
        if not result.get("node_count"):
            raise HTTPException(status_code=404, detail=f"无该 trace 的打点数据: {trace_id}")
        return TraceResponse(**result)

    @app.get("/metrics", response_model=MetricsResponse)
    async def get_metrics() -> MetricsResponse:
        """指标快照：各指标的 count / mean / p50 / p95 / max（进程内聚合）。"""
        return MetricsResponse(metrics=snapshot_metrics())

    # ==================== 图谱查询 ====================
    @app.get("/graph/entity/{entity_id}", response_model=GraphEntityResponse)
    async def get_graph_entity(
        entity_id: str,
        name: str = "",
        limit: int = 50,
        service: ReviewService = Depends(get_review_service),
    ) -> GraphEntityResponse:
        """查询实体画像（标签、属性）及其一跳关系（含折扣率、处罚金额等属性）。"""
        result = await run_in_threadpool(service.get_entity, entity_id, name, limit)
        return GraphEntityResponse(**result)

    @app.get("/graph/path", response_model=GraphPathResponse)
    async def get_graph_path(
        from_name: str,
        to_name: str,
        item_name: str = "",
        max_hops: int = 3,
        limit: int = 10,
        service: ReviewService = Depends(get_review_service),
    ) -> GraphPathResponse:
        """查询两个实体之间的多跳路径（如 品牌→授权店铺→商家），作为合规证据链。"""
        result = await run_in_threadpool(
            service.find_path, from_name, to_name, item_name, max_hops, limit
        )
        return GraphPathResponse(**result)

    # ==================== SSE 与任务状态 ====================
    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request):
        """SSE：实时推送审查各节点进度与最终结果。"""
        return StreamingResponse(
            sse_generator(task_id, request), media_type="text/event-stream",
        )

    @app.get("/status/{task_id}", response_model=ReviewTaskStatusResponse)
    async def get_task_status(
        task_id: str,
        service: ReviewService = Depends(get_review_service),
    ) -> ReviewTaskStatusResponse:
        return ReviewTaskStatusResponse(**service.get_status(task_id))

    @app.get("/health")
    async def health():
        return {"ok": True}


if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8002)
