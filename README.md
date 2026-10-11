# 电商合规审查知识库系统（Graph RAG）

> 基于 **LangGraph** + **知识图谱** + **混合检索** 的电商商品/商家合规审查系统。
>   
> 把商品详情页、商家资质档案、平台治理公告、违规处罚决定书等丢进去，自动切分、向量化、建知识图谱，
>   
> 再以「多路检索 → RRF 融合 → 重排 → 证据链 → 合规规则 → 人工复核 → 报告」的审查流水线，
>   
> 输出带证据链与风险等级的合规审查结论。

本系统面向**电商平台治理**场景（违禁词/极限词宣传、资质缺失或过期、违规未整改、
问题货源、类目错放与价格违规、品牌授权链穿透等），把口语化的审查问题变成
**有依据、可追溯、可复核**的审查结论。

> **项目定位**：本仓库是「电商平台合规审查」场景的**参考实现 / 演示项目**。
> 其中的示例文档、商家名称、经营指标与评测样本均为**合成的演示数据**，不含任何真实商家或个人信息。
> 它完整展示了从文档导入建图、多路混合检索到规则审查 + 人工复核的工程链路与关键取舍，
> 可作为技术方案参考；接入真实业务前，请自行完成合规、安全与性能评估。

---

## 功能特性

### 文档导入（`:8000`）

- 支持 **Markdown / HTML / Word(docx) / PDF**；HTML/Word 由内置转换器统一转为 Markdown，**保留表格与章节结构**
- 按**章节 / 条款 / 段落**语义切分，chunk 自动注入业务元数据（`company` / `doc_type` / `publish_date` / `source` / `chapter` / `clause_no`）
- LLM 识别文档所属**主体名（商家/店铺）**，作为知识库的隔离与过滤维度（`item_name`）
- LLM 按电商实体/关系词表抽取，写入 **Neo4j** 构建知识图谱（含**折扣率 / 违禁词 / 资质有效期 / 处罚金额**等关系属性）
- 密集 + 稀疏**双向量**写入 **Milvus**，支持混合检索

### 智能查询（`:8001`）

- **主体名确认**：口语化提问先对齐到知识库里的标准主体名，避免"检索跑偏"
- **多路混合检索**并行执行：BGE-M3 向量检索、HyDE 假设性文档检索、知识图谱多跳遍历
- **RRF 融合**（Reciprocal Rank Fusion）+ **BGE-Reranker 精排**，多路结果统一排序
- **SSE 流式输出**：逐 token 推送 `delta`，实时回传节点进度
- **多轮会话**：历史对话存 MongoDB，支持代词指代消解与 `item_names` 回填

### 合规审查（`:8002`）

- **8 节点审查流水线**：意图识别 → 查询改写/HyDE → 多路检索 → RRF 融合与重排 →
  
  证据链构建 → **合规规则检查** → **人工复核（HITL）** → 报告生成
- **内置合规规则引擎**（纯 Python、阈值可配）：
  | 规则 | 名称        | 判定                              |
  | -- | --------- | ------------------------------- |
  | R1 | 违禁词/极限词宣传 | 商品宣传命中违禁词 / 极限词（≥ 阈值个数）        |
  | R2 | 资质缺失或过期   | 受管控类目所需资质缺失，或有效期早于「今天 + 预警天数」   |
  | R3 | 违规未整改     | 近 N 年存在违规处罚但未见整改（无已整改留痕）        |
  | R4 | 问题货源      | 上游供应商为被投诉主体                     |
  | R5 | 类目错放与价格违规 | 类目错放，或促销折扣率 ≥ 阈值（涉嫌虚构原价）        |
  | R6 | 品牌授权链穿透   | 存在多层级品牌授权链（结构性提示，低风险）           |
- **证据链**：图谱多跳路径 + 命中文档片段 + 图谱关系统一组织，每条结论均可溯源
- **人工复核（HITL）**：高风险 hit 时用 LangGraph `interrupt()` 暂停，经 `POST /review/{id}/approve` 复核后恢复执行并出报告
- **结构化审查报告**：审查对象与范围 / 审查结论 / 风险等级 / 合规规则命中与建议 / 证据链 / 图谱路径
- **可观测**：每次审查输出一条 `trace_id`，各节点耗时汇成瀑布（`GET /trace/{trace_id}`），
  
  分支级检索延迟与规则命中数进指标（`GET /metrics`）
- **检索并发 + 容错**：三路检索并行发起、单路超时/失败自动降级，不阻塞整条审查

---

## 领域模型

电商合规的实体与关系在 `knowledge/schema/commerce_domain.py` 中**单点定义**，

被导入侧（抽取清洗）、查询侧（实体抽取、图谱遍历）、审查侧（合规规则）与 API 层共同引用，避免口径不一致。

**实体标签（11 类）**：`Merchant` `Product` `Person` `Brand` `Authority`

`Violation` `Complaint` `Promotion` `Qualification` `Supplier` `Category`

**关系类型（13 类）**：`SELLS`（销售，含 price/listing_date）`OPERATES`（经营）

`AUTHORIZED_BY`（品牌授权，含 auth_type/level/expire_date）`BELONGS_TO`（归属类目，含 category_code/is_misplaced）

`PROMOTES`（促销宣传，含 discount_rate/banned_words/is_fake_price）`HOLDS_CERT`（持有资质，含 cert_type/expire_date/status）

`SANCTIONED_BY`（受到处罚，含 amount/date/doc_no/reason/status）`COMPLAINED_IN`（涉及投诉，含 case_no/cause/amount）

`SUPPLIES`（供应货源）`SAME_OWNER`（同一经营者）`SHARED_BRAND`（共用品牌授权）

`MENTIONED_IN` / `RELATED_TO`（系统内部）

> 说明：`item_name` 字段沿用知识库既有命名，语义为「主体名（商家/店铺）」，
>   
> 作为知识库的隔离与过滤维度（设计决策见 `docs/REFACTOR_PLAN.md`）。

---

## 技术架构

![技术架构图](docs/architecture.pdf)

*图：技术架构（`docs/architecture.pdf`，由 `docs/generate_diagrams.py` 生成）*

### 技术栈

| 类别     | 选型                                                           |
| ------ | ------------------------------------------------------------ |
| 后端框架   | FastAPI + Uvicorn                                            |
| 工作流引擎  | LangGraph（状态图 / 条件边 / 并行分支 / `interrupt` HITL / MemorySaver） |
| 向量数据库  | Milvus（稠密 + 稀疏混合检索，WeightedRanker 归一融合）                      |
| 图数据库   | Neo4j                                                        |
| 文档数据库  | MongoDB（会话历史）                                                |
| 对象存储   | MinIO                                                        |
| 嵌入模型   | BGE-M3（dense + sparse）                                       |
| 重排序模型  | BGE-Reranker-Large（FlagEmbedding）                            |
| 大模型    | DeepSeek（`langchain-deepseek`）+ DashScope 视觉模型               |
| PDF 解析 | MinerU                                                       |
| 前端     | 原生 HTML + JS（无构建步骤）                                          |

---

## 目录结构

三个流程各自是**自包含单元**：自己的 `api / core / front / services / schema / nodes`，只共享 `schema/`（领域模型）与 `utils/`。

```
graph-rag-knowledge-base/
├─ knowledge/
│  ├─ schema/
│  │  └─ commerce_domain.py         # ★ 电商领域模型（实体/关系词表）单一事实来源
│  ├─ processor/
│  │  ├─ import_process/            # 导入流程（:8000）
│  │  │  ├─ api/import_router.py
│  │  │  ├─ nodes/                  # 含 doc_convert_node（HTML/Word→Markdown）
│  │  │  ├─ prompt/                 # 电商实体/关系抽取提示词
│  │  │  └─ ...
│  │  ├─ query_process/             # 查询流程（:8001）
│  │  │  ├─ api/query_router.py
│  │  │  └─ ...
│  │  └─ review_process/            # ★ 合规审查流程（:8002）
│  │     ├─ api/review_router.py    # FastAPI 应用与路由
│  │     ├─ core/{deps,paths}.py
│  │     ├─ front/review.html       # 审查控制台
│  │     ├─ nodes/                  # 8 个审查节点 + main_graph.py
│  │     ├─ prompts/review_prompts.py
│  │     ├─ rules/compliance_rules.py   # ★ 合规规则引擎（纯 Python）
│  │     ├─ services/review_service.py
│  │     ├─ schema/review_schema.py
│  │     └─ base.py  config.py  exceptions.py  state.py
│  ├─ data/sample_docs/             # ★ 电商示例文档（商品详情/资质档案/处罚决定书）
│  ├─ eval/                         # ★ 评测集（55 条）+ 指标/运行器 + 并发基准脚本
│  ├─ test/compliance/              # ★ 离线测试套件
│  ├─ utils/                        # 共享工具（LLM/BGE/Milvus/Neo4j/SSE/任务/安全/可观测性）
│  ├─ .env.example
│  ├─ requirements.txt              # 运行依赖（torch cu118 / BGE / mineru 等）
│  ├─ requirements-ci.txt           # CI 最小依赖集（不含重型包，2 分钟跑完全部检查）
│  └─ requirements-dev.txt          # 开发依赖（= requirements.txt + pytest/ruff/pre-commit）
├─ start_import_8000.bat            # 一键启动导入服务
├─ start_query_8001.bat             # 一键启动查询服务
├─ start_review_8002.bat            # ★ 一键启动审查服务
├─ Dockerfile                       # 多阶段构建（CUDA runtime + 非 root + 健康检查）
├─ docker-compose.yml               # 4 依赖服务 + 3 应用服务，一键起全栈
├─ docker-compose.gpu.yml           # GPU 叠加清单（可选）
├─ .dockerignore
├─ .github/workflows/ci.yml         # CI：lint / 测试矩阵 / 评测 / 基准 / 容器校验
├─ docs/REFACTOR_PLAN.md            # 设计与实现说明（领域模型 / 图谱 / 规则 / 接口契约）
├─ docs/电商合规审查_业务需求与流程设计.md    # 业务需求、输入输出契约与流程设计（可追溯性矩阵）
├─ docs/*.pdf                       # 流程/架构图（由 docs/generate_diagrams.py 生成）
├─ pyproject.toml                   # pytest / ruff 配置
├─ .pre-commit-config.yaml          # 提交前钩子（ruff）
├─ LICENSE                          # MIT
└─ README.md
```

---

## 快速开始

### 0. 克隆仓库

```bash
git clone <仓库地址> graph-rag-knowledge-base
cd graph-rag-knowledge-base
```

### 1. 依赖服务

| 服务      | 默认地址              | 用途      |
| ------- | ----------------- | ------- |
| Milvus  | `127.0.0.1:19530` | 向量库     |
| Neo4j   | `127.0.0.1:7687`  | 知识图谱    |
| MongoDB | `127.0.0.1:27017` | 会话历史    |
| MinIO   | `127.0.0.1:9000`  | 图片/对象存储 |

### 2. 安装依赖

```bash
cd knowledge
python -m venv .venv
.venv\Scripts\activate          # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -r requirements.txt
```

> 需要 GPU 环境（BGE-M3 / Reranker / MinerU 都跑在 `cuda:0`）。

### 3. 配置

```bash
copy knowledge\.env.example knowledge\.env      # Windows
cp knowledge/.env.example knowledge/.env        # Linux/macOS
```

按实际情况填 `.env`，至少要有：

| 变量                                                                      | 说明                |
| ----------------------------------------------------------------------- | ----------------- |
| `DEEPSEEK_API_KEY`                                                      | 导入/查询/审查用的 LLM    |
| `MILVUS_URL`                                                            | Milvus 地址         |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD`                       | Neo4j 连接          |
| `MONGO_URL` / `MONGO_DB_NAME`                                           | 会话历史，不配则历史功能整体不可用 |
| `MINIO_ENDPOINT` / `MINIO_BUCKET_NAME` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | MinIO 地址、桶名与凭证    |
| `CHUNKS_COLLECTION` / `ITEM_NAME_COLLECTION` / `ENTITY_NAME_COLLECTION` | 三个集合名             |
| `BGE_RERANKER_LARGE`                                                    | 重排序模型本地路径         |

安全与合规审查相关（阈值/开关，均有默认值，见 `.env.example`）：

`API_KEY`、`CORS_ALLOW_ORIGINS`、`HITL_ENABLED`、`HITL_RISK_LEVELS`、`HITL_REVIEW_TYPES`、
`COMPLIANCE_BANNED_WORD_HIT`、`COMPLIANCE_CERT_EXPIRE_DAYS`、`COMPLIANCE_PENALTY_YEARS`、
`COMPLIANCE_FAKE_PRICE_RATIO`、`COMPLIANCE_PENETRATION_DEPTH`、`KG_MAX_PATH_HOPS`。

### 4. 启动

**推荐**：直接双击（脚本已自动处理工作目录、`PYTHONPATH` 与代理变量）

```
start_import_8000.bat     →  导入服务
start_query_8001.bat      →  查询服务
start_review_8002.bat     →  审查服务
```

**或手动**（须在**项目根目录**启动，且让 `knowledge` 包可被导入）：

```bash
PYTHONPATH=$(pwd) python -m uvicorn \
  knowledge.processor.review_process.api.review_router:create_app \
  --factory --host 127.0.0.1 --port 8002
```

> ⚠️ 首次启动约 60 秒（加载 torch / FlagEmbedding / pymilvus），看到 `Application startup complete` 才算就绪。
>
> ⚠️ 若本机开着系统代理，请先清掉 `HTTP_PROXY / HTTPS_PROXY`：gRPC 也会走代理，会导致 Milvus 连接报 502。

### 5. 打开前端

| 页面          | 地址                             |
| ----------- | ------------------------------ |
| 文档导入        | <http://127.0.0.1:8000/import> |
| 知识库问答       | <http://127.0.0.1:8001/>       |
| **合规审查控制台** | <http://127.0.0.1:8002/>       |
| 接口文档        | <http://127.0.0.1:8002/docs>   |

### 6. 容器化一键起全栈（可选）

不想手工装 Milvus / Neo4j / Mongo / MinIO 时，用 Compose 一次起 4 个依赖 + 3 个应用：

```bash
# 起全栈（首次构建镜像较久：torch cu118 等依赖约 10GB）
docker compose up -d --build

docker compose ps                          # 看健康状态（全部 healthy 才算起好）
docker compose logs -f app-review          # 跟踪审查服务日志
```

| 服务 | 地址 | 说明 |
| -- | -- | -- |
| `app-import` | <http://127.0.0.1:8000/import> | 文档导入建图 |
| `app-query` | <http://127.0.0.1:8001/> | 知识库问答 |
| `app-review` | <http://127.0.0.1:8002/> | 合规审查控制台（`/docs` 可访问） |
| Milvus / Neo4j / MinIO / Mongo | 19530 / 7474 / 9001 / 27017 | 依赖服务，端口均已映射，便于本机排查 |

**关于 GPU**：镜像基于 `nvidia/cuda:11.8.0-runtime-ubuntu22.04`（与 `torch==2.7.1+cu118` 对齐），
但 Compose 里**默认不预留 GPU**，所以没有显卡的机器也能一把起（torch 回落 CPU，功能完整、推理变慢）。
有 NVIDIA 容器运行时（nvidia-container-toolkit）时叠加 GPU 清单：

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

密钥与模型配置仍走 `knowledge/.env`（文件缺失时不报错，用演示默认值启动）；
Compose 只覆盖**容器内网地址**（`MILVUS_URL=http://milvus:19530` 等），避免 .env 里的 `127.0.0.1` 连到容器自己。

---

## 容器化与 CI

| 文件 | 作用 |
| -- | -- |
| `Dockerfile` | **多阶段 + CUDA runtime + 非 root 运行 + 健康检查**；依赖层与代码层分离，改业务代码不必重装 torch |
| `docker-compose.yml` | 4 依赖 + 3 应用，全部带健康检查；应用服务用 `condition: service_healthy` 等待依赖就绪 |
| `docker-compose.gpu.yml` | GPU 叠加清单：给三个应用服务加 `nvidia` 设备预留 |
| `.dockerignore` | 排除 `.env`（密钥绝不进镜像层）、虚拟环境、缓存、运行期产物 |
| `knowledge/requirements-ci.txt` | **CI 用的最小依赖集**：不含 torch / mineru，2 分钟跑完全部检查 |
| `.github/workflows/ci.yml` | push / PR 自动跑 lint、多版本测试、评测指标、并发基准、容器配置校验 |

CI 的 4 个 job：

| job | 内容 | 说明 |
| -- | -- | -- |
| `lint` | `ruff check .` | 静态检查单独成 job，失败时不必等测试跑完 |
| `test` | Python 3.10 / 3.12 矩阵：`pytest` + `run_eval.py` + 并发基准 | 评测分与并发收益都进 CI——**回退即拦** |
| `docker` | `docker compose config -q`（含 GPU 叠加）+ `docker buildx build --check` | 只做配置与 Dockerfile 校验，不拉 10GB 镜像 |
| `gpu-smoke` | 起全栈 → 三服务 `/health` → `/docs` → 提交一次审查 | **默认跳过**：托管 runner 无 GPU，需自备 `self-hosted,gpu` runner 并设仓库变量 `ENABLE_GPU_RUNNER=true` |

---

## 核心流程

### 导入流程（9 节点）

![导入流程图](docs/import_flow.pdf)

### 审查流程（8 节点）

![审查流程图](docs/review_flow.pdf)

设计要点：

- **并行分支只返回自己修改的字段**（如 `return {"embedding_chunks": ...}`），否则 LangGraph 会抛
  
  `InvalidUpdateError: Can receive only one value per step`。
- **SSE 三层解耦**：`task_util` 只管任务状态、`sse_util` 只管消息推送、`BaseNode` 是唯一同时知道两者的地方。
- **HITL 控制流异常必须穿透**：`review_process/base.py` 的 `is_control_flow_exception` 保证
  
  `langgraph` 抛出的 `GraphInterrupt` 不被节点基类包装，否则中断/恢复机制失效。

---

## 性能与可靠性（工程化加固）

LLM 应用的两类典型故障——**慢**（检索串行 + 图谱 N+1）与**不可解释**（出错了只能翻日志）——
在本项目中都有对应的实现与量化数据。

### 检索并发化 + 图谱 N+1 消除

**问题**：三路检索（Milvus 向量 / Milvus 稀疏 / Neo4j 图谱）原本在一个节点内顺序执行，
墙钟 = sum(三路)；图谱分支内部对「每个实体对」各开一次 Neo4j session（N+1 往返），
实体数一多就是 n + C(n,2) 次跨进程往返。

**动作**：三路检索改为线程池并发（`ThreadPoolExecutor` + `as_completed`），
每路独立超时（`REVIEW_SEARCH_BRANCH_TIMEOUT`，默认 20s）、独立容错；
图谱分支改用 `UNWIND $names` / `UNWIND $pairs` 批量下推，
用子查询内的 `LIMIT $per_limit` 保持「每个种子各自的条数上限」这一原语义。

**验证**（`python knowledge/eval/bench_search_concurrency.py`，输出为实测值）：

| 方案 | 单路耗时 向量/稀疏/图谱 | 墙钟(ms) | 相对串行 |
| --- | --- | --- | --- |
| 串行（整改前） | 400/300/900 | ≈1600 | 1.00x |
| 并发（整改后） | 400/300/900 | ≈900 | **≈1.8x** |

| 方案 | 往返次数 | 墙钟(ms) | 提速 |
| --- | --- | --- | --- |
| 逐实体 + 逐实体对（整改前, N+1） | 4 | ≈200 | 1.00x |
| UNWIND 批量下推（整改后） | 2 | ≈100 | **≈2.0x**（往返数减半，省约 50% 耗时） |

> 表中为一次代表性实测值（墙钟随机器/负载略有浮动，往返次数是确定值）；
> 单路耗时与 RTT 是脚本参数（模拟跨机房网络），衡量的是**调度与往返结构**的差异，
> 换成线上实测值重跑即可：`--branch-ms 500,320,860 --rtt 80 --repeat 5`。

**容错边界**（`knowledge/test/compliance/test_concurrency.py` 逐条覆盖）：

| 场景 | 行为 |
| --- | --- |
| 单路抛异常 | 该路降级为空结果，其余两路结果照常返回 |
| 单路超时 | 到点放弃等待（`pool.shutdown(wait=False)`），标记 timeout 后继续走流程 |
| 三路全挂 | 返回完整空结构，节点契约不变、不抛异常（离线可跑的前提） |

### 链路追踪与指标

每个节点由 `BaseNode` 统一打点（`knowledge/utils/observability.py`），
`trace_id` 随 state 在图中透传并注入日志前缀，于是「同一次审查」的所有日志可一条 grep 捞出：

```
[trace=rv-demo-2024-0512] 节点耗时瀑布（总计 14 ms）
   1. 意图识别         ███████████████                   4.2 ms (29.5%)  intent=compliance
   2. 查询改写         ███████                           1.9 ms (13.3%)
   3. 多路检索         ████████████████████████████      7.5 ms (52.5%)  vec=0 kg=0 triples=0 paths=0 sparse=0 branches=sparse:6|kg:5|vector:2 wall_ms=6
   4. 融合重排         █                                 0.2 ms ( 1.5%)  rrf=0 docs=0
   5. 证据链构建        █                                 0.1 ms ( 0.6%)  evidence=0
   6. 合规规则检查       █                                 0.2 ms ( 1.5%)  findings=0 risk=无
   7. 人工复核         █                                 0.1 ms ( 1.1%) ✅interrupted
```

- **默认纯内存**（环形缓冲 `OBSERVABILITY_MAX_TRACES`，默认 200 条），离线测试零副作用；
  需要落库时用 `OBSERVABILITY_SINK=jsonl|mongo`，**持久化失败一律降级为日志**，不影响主流程。
- HITL 的 `interrupt()` 会被识别为 `status=interrupted` 而非 error——否则「正常暂停」和「真出错」在瀑布里分不清。

### LLM 作为「不可靠依赖」的处理

LLM 是外部依赖，会超时、会返回带 ``` 围栏的 JSON、会在长文档上截断。统一收敛在
`knowledge/utils/llm_client.py`：

- **超时 + 指数退避重试**：只对瞬时错误（timeout / connection reset / 429 / 5xx / overloaded）重试，
  业务错误立即失败（避免无意义重试放大延迟）；
- **JSON 清洗单点化**：`parse_json_response` 依次尝试「原文直解 → 去代码围栏 → 括号配平抽取」，
  失败返回调用方的默认值。原先散落在 4 个节点里的正则清洗全部收敛到这里（一处修、四处受益）；
- **token 计数**：从 `usage_metadata` 记录 `last_usage`，为后续成本观测留口子。

---

## 接口一览

### 审查服务 `:8002`

| 方法   | 路径                            | 说明                                                  |
| ---- | ----------------------------- | --------------------------------------------------- |
| POST | `/review`                     | 提交审查（商家名、审查类型、问题描述）；同步返回结果或 `task_id`               |
| GET  | `/review/{review_id}`         | 获取审查结果（证据链、风险等级、findings、报告、`trace_id`）              |
| POST | `/review/{review_id}/approve` | 人工审批（通过/驳回），恢复被 `interrupt` 暂停的图                    |
| GET  | `/trace/{trace_id}`           | 节点耗时瀑布（每节点耗时/状态/产出摘要），用于定位「慢在哪一步」                  |
| GET  | `/metrics`                    | 指标快照：节点耗时、检索分支延迟、召回条数、规则命中数（count/mean/p50/p95/max） |
| GET  | `/graph/entity/{entity_id}`   | 查询实体画像及其一跳关系（含关系属性）                                 |
| GET  | `/graph/path`                 | 查询两个实体之间的多跳路径（`from_name` / `to_name` / `max_hops`） |
| GET  | `/stream/{task_id}`           | SSE：`ready` / `progress` / `final`                  |
| GET  | `/status/{task_id}`           | 轮询任务状态与已完成节点（中文名）                                   |
| GET  | `/` `/review.html`            | 审查控制台前端                                             |

**提交审查示例**

```bash
curl -X POST http://127.0.0.1:8002/review -H "Content-Type: application/json" -d '{
  "company": "广州XX贸易有限公司",
  "review_type": "product_compliance",
  "question": "该商品标题使用“最安全”，且店铺未取得食品经营许可证，是否合规？",
  "is_stream": false
}'
```

### 查询服务 `:8001`

| 方法   | 路径                      | 说明                                                |
| ---- | ----------------------- | ------------------------------------------------- |
| POST | `/query`                | 提交查询；`is_stream=false` 同步返回答案，`true` 返回 `task_id` |
| GET  | `/stream/{task_id}`     | SSE 长连接                                           |
| GET  | `/status/{task_id}`     | 轮询任务状态与进度                                         |
| GET  | `/history/{session_id}` | 拉取会话历史                                            |
| GET  | `/chat.html` `/`        | 聊天前端                                              |

### 导入服务 `:8000`

| 方法   | 路径                  | 说明                       |
| ---- | ------------------- | ------------------------ |
| POST | `/upload`           | 上传文档，返回 `task_id`，后台跑导入图 |
| GET  | `/status/{task_id}` | 查询导入进度                   |
| GET  | `/import`           | 上传前端                     |

---

## 评测与测试

### 示例文档

`knowledge/data/sample_docs/` 提供三份虚构示例（商品详情页 HTML / 商家资质档案 / 平台违规处罚决定书），

覆盖 R1/R2/R3 的取证场景。详见该目录下 README。

### 评测集

| 文件                               | 说明                                                         |
| -------------------------------- | ---------------------------------------------------------- |
| `knowledge/eval/qa_pairs.jsonl`  | **55 条**样本：主体 / 审查类型 / 问题 / 期望意图 / 期望规则 / 期望风险等级 / HITL / 关系 / 图谱路径 / 经营指标 |
| `knowledge/eval/dataset.py`      | 加载器 + 相对日期解析 + 指标计算（规则命中 F1 / 风险等级 / HITL / 意图）             |
| `knowledge/eval/run_eval.py`     | 评测运行器：一键出分，可选 LLM-as-Judge                                |
| `knowledge/eval/evaluation.json` | 数据集元信息（含分布统计与期望分数；由数据重算，测试会校验不漂移）                          |

数据分布：8 类审查类型全覆盖，正样本 **34** 条（覆盖 R1–R6）/ 负样本 **21** 条（度量误报），风险等级 高 15 / 中 15 / 低 4 / 无 21。

**日期字段用相对 token 而非绝对日期**（`@today` / `@-30d` / `@+365d` / `@-2y`），
读取时才落到绝对日期——否则「两年前的处罚」这类样本会随着时间推移被挤出 R3 的三年窗口，
评测分数会莫名下降（数据集腐烂）。

```bash
python knowledge/eval/run_eval.py                  # 离线出分（无需 Milvus/Neo4j/LLM）
python knowledge/eval/run_eval.py --judge          # 追加 LLM-as-Judge（无可用 LLM 时自动跳过）
python knowledge/eval/run_eval.py --json out.json  # 同时输出机器可读结果
```

实测分数（`python knowledge/eval/run_eval.py`，可复现）：

| 指标 | 分数 | 口径 |
| -- | -- | -- |
| `rule_hit_f1`（micro） | **100%**（P 100% / R 100%） | 规则命中集合与期望完全一致；分规则召回 R1–R6 均 100% |
| `risk_level_accuracy` | **100%** | 整体风险等级（取命中规则的最大严重度）完全一致 |
| `hitl_accuracy` | **100%** | HITL 触发判定与 `compliance_check_node` 口径一致 |
| `intent_accuracy`（离线关键词兜底） | **69.1%** | **参考值**：在线由 LLM 决定意图；该分数如实反映兜底规则的短板 |

意图兜底的主要偏差（运行器会直接打印，用于说明「哪类问题必须依赖 LLM」）：

```
compliance→fact 9 例；fact→compliance 3 例；path→fact 3 例；compliance→path 1 例；path→compliance 1 例
```

> 前两项/第三项是**确定性结论**，测试里按满分强断言（`test_eval_dataset.py`）；
> 意图准确率只做 0.6 下限保护，避免为了「看起来好看」而把数据集写成关键词规则的练习册。

**LLM-as-Judge**（在线能力，可跳过硬失败）：把「系统给出的结论 + 证据」与参考期望一起送 LLM 打分（1–5 分，
并要求列出缺乏证据支撑的结论），用于评估报告质量这类无法用规则判定的维度。
无 API Key / 未装 `langchain_deepseek` 时打印「已跳过」并正常退出，不会让离线环境变红；
也可用 `--reports gen.json`（`{qa_id: 报告文本}`）把真实生成的报告送评。

### 离线测试套件

```bash
pip install -r knowledge/requirements-dev.txt   # 或 pip install pytest fastapi uvicorn httpx
pytest knowledge/test/compliance -q             # 全部（137 passed / 8 skipped）
```

覆盖领域模型、抽取清洗、图谱查询（含批量下推）、合规规则、审查流程（HITL）、API、
评测集与运行器、LLM 客户端加固、可观测性、**多路检索并发调度**（并发齐发/墙钟收益/分支超时/异常隔离），
以及**部署配置**（Dockerfile 多阶段与非 root、compose 健康检查与依赖顺序、GPU 叠加、CI job 与最小依赖集）。

Milvus / Neo4j / LLM **均非必需**：检索/改写/生成节点在依赖不可用时有模板兜底，

图谱工具在无驱动时返回空结果而非报错；`test_extraction.py` / `test_hardening.py` 中依赖重型三方库的用例

在精简环境会自动 skip。详见 `knowledge/test/compliance/README.md`。

---

## 常见问题

| 现象                                                                                   | 原因 / 处理                                                     |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| 启动很慢                                                                                 | 正常，首次导入要拉起 torch 等重型库，约 60s                                 |
| Milvus 报 `illegal connection params...` 且日志有 `HTTP proxy returned response code 502` | 系统代理劫持了 gRPC，清掉 `HTTP_PROXY/HTTPS_PROXY` 再启动                |
| 请求返回 401                                                                             | 设置了 `API_KEY` 但请求未带 `X-API-Key` 请求头；本地演示可留空 `API_KEY`      |
| 浏览器报 CORS 错误                                                                        | 前端域名不在 `CORS_ALLOW_ORIGINS` 白名单；该白名单不含 `*`，需显式添加            |
| 问答返回「抱歉，我无法识别您询问的具体主体名称」                                                             | 降级话术：LLM 没抽到主体名，或 Milvus 连不上。先确认向量库在跑、集合已导入数据               |
| 审查一直停在「待人工复核」                                                                        | 高风险命中了 HITL（预期行为）：调用 `POST /review/{id}/approve` 复核后才会出报告   |
| 审查报告里证据链为空                                                                           | 离线/未接图谱时，证据链会回退用注入的关系作为「关系型证据」；真实场景请确认 Milvus/Neo4j 已导入数据   |
| 进度条一直显示英文节点名                                                                         | `task_util.py` 的 `_NODE_NAME_TO_CN` 键名要与 `BaseNode.name` 一致 |
| `/history` 返回 500                                                                    | MongoDB 未配置；`MONGO_URL` / `MONGO_DB_NAME` 缺失                |
| 首次 `docker compose up --build` 很慢                                                   | 预期：镜像要装 torch cu118 / sentence-transformers 等约 10GB 依赖，之后重建命中缓存    |
| compose 里应用容器起不来 / 连不上 Milvus                                                          | 容器内必须用服务名互访（`http://milvus:19530`）。compose 已覆盖这些地址；若自行改 command 或另加环境变量需同步 |
| 加了 `docker-compose.gpu.yml` 后起不来                                                        | 宿主机需装 nvidia-container-toolkit 并能 `docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`；否则不要叠加该文件 |

---

## 说明

本项目的架构方案（LangGraph + 多路混合检索 + 知识图谱）在设计阶段针对**电商合规审查**场景
独立设计与实现：领域词表、抽取/审查提示词、合规规则库、示例数据、评测集与前端文案
均为本项目编写；技术组件选型与图结构沿用同一套骨架，三大能力域（导入建图 / 混合检索 /
审查流水线 + HITL）完整落地。

许可证与上游授权情况见 `LICENSE`。

---
