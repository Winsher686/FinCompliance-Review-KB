"""统一安全配置：CORS 白名单 + API Key 鉴权（三个服务单元共用）。

设计取舍（电商合规扩展 / 场景适配）：
1. **CORS 不再使用 ``*``**：白名单由环境变量 ``CORS_ALLOW_ORIGINS`` 控制，
   默认仅放行本机三个服务端口。``*`` 与 ``allow_credentials=True`` 同时使用
   会被浏览器视为不安全，且等于对所有站点开放。
2. **API Key 采用「按需开启」**：设置 ``API_KEY`` 后，除公开路径外全部接口
   强制校验请求头 ``X-API-Key``；不设置则放行（保持离线 / 演示环境可直接启动），
   并在启动日志给出告警。生产环境务必设置 ``API_KEY``。
3. 鉴权中间件用**原生 ASGI** 实现，而不是 ``BaseHTTPMiddleware``——后者会包裹
   响应对象，可能影响 ``/stream`` 的 SSE 流式输出。
"""

import logging
import os
from typing import List

from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# 无需鉴权的路径：健康检查、接口文档、静态前端
_PUBLIC_PATHS = {"/", "/health"}
_PUBLIC_PREFIXES = ("/front", "/docs", "/redoc", "/openapi.json")
_PUBLIC_SUFFIXES = (".html",)

_DEFAULT_ORIGINS = (
    "http://127.0.0.1:8000,http://127.0.0.1:8001,http://127.0.0.1:8002,"
    "http://localhost:8000,http://localhost:8001,http://localhost:8002"
)


def get_cors_origins() -> List[str]:
    """从环境变量读取 CORS 白名单；未配置时回退到本机三端口。"""
    raw = os.getenv("CORS_ALLOW_ORIGINS", _DEFAULT_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _is_public(path: str) -> bool:
    if path in _PUBLIC_PATHS:
        return True
    if path.endswith(_PUBLIC_SUFFIXES):
        return True
    return any(path.startswith(prefix) for prefix in _PUBLIC_PREFIXES)


class ApiKeyMiddleware:
    """原生 ASGI 鉴权中间件：``API_KEY`` 为空时不拦截。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        api_key = os.getenv("API_KEY", "")
        path = scope.get("path", "")
        # 预检请求（OPTIONS）不携带自定义头，必须放行交给 CORS 处理
        if not api_key or scope.get("method") == "OPTIONS" or _is_public(path):
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        provided = headers.get(b"x-api-key", b"").decode("latin-1")
        if provided != api_key:
            response = JSONResponse(
                {"detail": "unauthorized: missing or invalid X-API-Key"},
                status_code=401,
            )
            return await response(scope, receive, send)

        return await self.app(scope, receive, send)


def install_security(app) -> None:
    """给 FastAPI 实例安装 CORS 白名单 + API Key 鉴权（三个服务统一调用）。

    中间件添加顺序说明：``add_middleware`` 后加的更靠外层，因此先加鉴权、
    后加 CORS，保证 CORS 最外层、能正确处理跨域预检。
    """
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(ApiKeyMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if not os.getenv("API_KEY", ""):
        logger.warning(
            "API_KEY 未配置：接口鉴权已关闭。仅建议本地 / 演示环境使用，"
            "生产环境请设置 API_KEY 环境变量。"
        )
