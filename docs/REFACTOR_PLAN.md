# 电商合规审查系统 · 设计与实现说明

> 本系统面向**电商商品/商家合规审查**，由知识入库、混合检索、规则审查（含人工复核）三条链路构成。
> **技术栈**：Python 3.10+ / FastAPI / LangGraph / Milvus(稠密+稀疏) / Neo4j / BGE-M3 / HyDE / RRF / BGE-Reranker / SSE / MongoDB / MinIO，后端全部为 Python。

---

## 0. 技术骨架基线

| 项 | 形态 |
|---|---|
| 导入流程 `:8000` | `entry → pdf_to_md / md_img → document_split → item_name_node → bge_embedding_chunks → import_milvus → kg_graph` |
| 查询流程 `:8001` | `item_name_confirm → 四路并行(向量 / HyDE / KG一跳 / MCP联网) → join → rrf → rerank → answer_output` |
| 核心维度 | `item_name`（主体名）贯穿 Milvus 三集合、Neo4j `Entity{name,item_name}` / `Chunk`、全部 Prompt、Mongo、前端 |
| 图谱 | 通用 `Entity` 动态标签（未按业务域细分的标签体系），关系白名单 10 种，**只有一跳查询、关系无属性** |
| 检索 | BGE-M3 dense+sparse → Milvus `WeightedRanker`；RRF 融合；BGE-Reranker 精排；SSE 流式 |
| 场景文案 | 提取 Prompt、HyDE、答案 Prompt、前端标题均为通用文档问答口吻，未针对业务场景 |

---

## 1. 总体设计策略（4 条主线）

1. **领域层**：电商合规词表（实体 / 关系 / 属性），图谱 schema 细化为「商家–商品–资质–违规–投诉–货源–品牌」。
2. **抽取层**：导入侧实体/关系抽取 Prompt（含电商 few-shot），Chunk 携带业务元数据（主体名 / 文档类型 / 发布日期 / 来源）。
3. **审查链路**：**`review_process` 自包含单元**承载合规审查流水线（意图识别 → 检索 → 证据链 → 规则 → HITL → 报告）；`query_process` 承载「合规问答」。
4. **契约层**：`/review`、`/graph/*` 接口 + SSE，与既有接口向后兼容。

**架构约束**：Milvus / Neo4j / MinIO / MongoDB 的选型，RRF / HyDE / Reranker / BGE-M3 的使用方式，SSE 三层解耦结构，`BaseNode` 的进度上报机制。

---

## 2. 领域模型设计（`knowledge/schema/commerce_domain.py`）

领域层的**单一事实来源**。导入侧（抽取清洗）、查询侧（实体抽取、图谱遍历）、审查侧（合规规则）
与 API 层全部引用本模块，禁止在别处硬编码词表。

### 2.1 核心实体（Neo4j Label / Milvus entity label，中文映射）

| 中文 | Label | 关键属性 |
|---|---|---|
| 商家/店铺 | `Merchant` | uscc(统一社会信用代码), shop_id, reg_address, business_scope |
| 商品 | `Product` | sku, price, brand |
| 自然人 | `Person` | id_no_masked |
| 品牌 | `Brand` | brand_owner, trademark_no |
| 监管/平台治理机构 | `Authority` | level |
| 违规记录 | `Violation` | amount(处罚金额), date, doc_no, reason, status |
| 投诉纠纷 | `Complaint` | case_no, cause, amount, handle_result |
| 促销宣传 | `Promotion` | amount, date, discount_rate, banned_words, is_fake_price |
| 资质证照 | `Qualification` | cert_no, cert_type, expire_date, status |
| 供应商 | `Supplier` | role, product |
| 商品类目 | `Category` | category, code, is_regulated |

### 2.2 关系类型（13 类，含关系属性）

| 中文 | type | 方向 | 关系属性 |
|---|---|---|---|
| 销售 | `SELLS` | Merchant→Product | price, listing_date |
| 经营 | `OPERATES` | Person→Merchant | role, since |
| 品牌授权 | `AUTHORIZED_BY` | Brand→Merchant | auth_type, expire_date, level |
| 归属类目 | `BELONGS_TO` | Product→Category | category_code, is_misplaced |
| 促销宣传 | `PROMOTES` | Merchant→Promotion | amount, date, discount_rate, banned_words, is_fake_price |
| 持有资质 | `HOLDS_CERT` | Merchant→Qualification | cert_no, cert_type, expire_date, status |
| 受到处罚 | `SANCTIONED_BY` | Merchant/Product→Authority | amount, date, doc_no, reason, status |
| 涉及投诉 | `COMPLAINED_IN` | Merchant/Product/Supplier→Complaint | case_no, cause, amount |
| 供应货源 | `SUPPLIES` | Supplier/Merchant→Merchant | product, amount, direction |
| 同一经营者 | `SAME_OWNER` | Merchant↔Merchant | — |
| 共用品牌授权 | `SHARED_BRAND` | Merchant/Brand→Merchant | brand, ratio |
| 提及于 | `MENTIONED_IN` | Entity→Chunk | （保留，查询侧反查 chunk 依赖） |
| 相关 | `RELATED_TO` | *→* | （兜底） |

### 2.3 文档类型与合规维度

- `DOC_TYPES`：`violation_decision` / `merchant_profile` / `product_page` / `platform_notice` /
  `promotion_record` / `customer_complaint` / `inspection_report` / `other`，
  由文件名与正文关键词自动推断（`infer_doc_type`）。
- `COMPLIANCE_DIMENSIONS`：商品宣传用语、类目与价格、资质证照、违规处罚与整改、投诉纠纷、
  货源供应链、品牌授权链。改写节点通过 `compliance_dimensions_text()` 派生，**不另写硬编码词表**。

---

## 3. 文件清单与实现要点

### 3.1 配置文件

| 文件 | 实现要点 |
|---|---|
| `knowledge/.env.example` | 统一模板：大模型、Milvus/Neo4j/Mongo/MinIO、`API_KEY`/`CORS_ALLOW_ORIGINS`、`COMPLIANCE_*` 规则阈值、图谱多跳深度、HITL 开关与审查类型 |
| `knowledge/processor/import_process/config.py` | 文档类型配置、业务元数据字段、实体名长度上限（15→40，商家全称较长） |
| `knowledge/processor/query_process/config.py` | 主体名对齐阈值、检索规模、图谱路径遍历深度 |
| `knowledge/processor/review_process/config.py`（新） | 审查单元独立配置（含 HITL 触发策略） |

### 3.2 领域模型与图谱

| 文件 | 实现要点 |
|---|---|
| `knowledge/schema/commerce_domain.py`（新） | 电商实体 / 关系 / 属性常量与中文映射，供抽取、清洗、API 共用 |
| `knowledge/processor/import_process/nodes/kg_graph_node.py` | `ALLOWED_ENTITY_LABELS`/`ALLOWED_RELATION_TYPES` 改引用领域层；MERGE 语句支持**关系属性**；实体名长度上限放宽 |
| `knowledge/processor/query_process/nodes/kg_search_node.py` | Cypher 查询：支持**多跳路径**（如 品牌→授权店铺→商家）、返回关系属性、路径作为证据 |
| `knowledge/utils/neo4j_util.py` | 新增图谱查询函数：`find_path_between(from,to,max_hops)`、`get_entity_relations(entity_id)`、`get_entity_profile(entity_id)` |
| `knowledge/schema/__init__.py` | 导出领域模块与 `COMPLIANCE_DIMENSIONS` |

### 3.3 抽取与业务 Prompt

| 文件 | 实现要点 |
|---|---|
| `knowledge/processor/import_process/prompt/knowledge_graph_prompt.py` | **全量重写**：电商实体/关系抽取 + 电商 few-shot（商品详情/资质档案/处罚决定书示例） |
| `knowledge/processor/import_process/prompt/item_name_prompt.py` | 主体名识别 → **商家/店铺名识别** |
| `knowledge/processor/review_process/prompts/review_prompts.py` | `REVIEW_TYPE_CN` 改电商审查类型；意图识别/改写/HyDE/报告 4 组 Prompt 全部改写 |
| `knowledge/processor/query_process/prompts/query_prompt.py` | `ENTITY_EXTRACT_SYSTEM_PROMPT`→电商实体；`ANSWER_PROMPT`→合规审查口吻（结论/依据/风险等级/建议） |
| `knowledge/processor/query_process/prompts/item_name_extract_prompt.py` | HyDE 假设文档 → **电商合规假设性文档**（宣传用语/类目价格/资质/违规/投诉/货源/授权维度） |

### 3.4 文档处理与导入链路

| 文件 | 实现要点 |
|---|---|
| `knowledge/processor/import_process/nodes/entry_node.py` | 支持 PDF / HTML / Word（`.pdf/.html/.htm/.doc/.docx/.md`），补全文档元数据 |
| `knowledge/processor/import_process/nodes/doc_convert_node.py` | HTML / Word → Markdown（保留表格），对接 MinerU 分支 |
| `knowledge/processor/import_process/nodes/document_spliter_node.py` | 按**章节 / 条款 / 段落**分块，chunk 注入元数据（主体名 / 文档类型 / 发布日期 / 来源 / 章节路径 / 条款号） |
| `knowledge/processor/import_process/nodes/import_milvus_node.py` | 双向量写入 Milvus；**按主体增量重建**（不再 drop_collection 整表清空） |
| `knowledge/processor/import_process/services/import_file_service.py` | 上传文件名净化（防路径遍历 + 扩展名白名单） |
| `knowledge/processor/import_process/state.py` | 新增业务元数据字段（doc_type / publish_date / source / company） |

### 3.5 检索与查询链路

| 文件 | 实现要点 |
|---|---|
| `knowledge/processor/query_process/state.py` | 新增：`intent`、`kg_paths`、`evidence_chain` 等字段 |
| `knowledge/processor/query_process/nodes/item_name_confirm_node.py` | **主体确认加强**（商家名对齐 + 阈值分档 + 反问澄清） |
| `knowledge/processor/query_process/nodes/hyde_search_node.py` | 电商 HyDE 假设性文档 |
| `knowledge/processor/query_process/nodes/vector_search_node.py` | 主体维度过滤（`item_name`），保留 dense+sparse |
| `knowledge/processor/query_process/nodes/mcp_search_node.py` | 输出作为「外部证据」标注；密钥改由环境变量提供 |
| `knowledge/processor/query_process/nodes/rerank_node.py` | 保留（BGE-Reranker 精排 + 分数落差动态截断） |
| `knowledge/processor/query_process/main_graph.py` | 意图路由（事实查询 / 路径查询 / 合规检查 / 报告生成） |

### 3.6 LangGraph 工作流 + 审查节点（review_process）

| 文件 | 实现要点 |
|---|---|
| `knowledge/processor/review_process/`（**新单元**） | 完整自包含：`api / core / front / nodes / prompts / rules / services / schema / state.py / config.py / base.py` |
| `nodes/intent_recognize_node.py` | 意图识别（fact / path / compliance / report），LLM 优先、关键词规则兜底 |
| `nodes/query_rewrite_node.py` | 查询改写 + HyDE（维度串来自领域层） |
| `nodes/multi_search_node.py` | 并行：向量 / BM25(稀疏) / 图谱多跳 |
| `nodes/rrf_rerank_node.py` | RRF 融合 + BGE-Reranker（复用现有实现） |
| `nodes/evidence_chain_node.py` | 证据链构建（图谱路径 + 文档片段 → 结构化证据，并归一化关系供规则引擎使用） |
| `nodes/compliance_check_node.py` | 合规规则检查（调用规则库） |
| `nodes/hitl_node.py` | 人工复核：高风险或指定审查类型触发 `interrupt()`，等待 `/approve` |
| `nodes/report_generate_node.py` | 结构化审查报告（结论 / 证据链 / 风险等级 / 建议） |
| `rules/compliance_rules.py` | 内置规则库（见 §6），纯 Python、离线可测 |

### 3.7 API 与 Schema

| 文件 | 实现要点 |
|---|---|
| `knowledge/processor/review_process/api/review_router.py`（新） | `/review`、`/review/{id}`、`/review/{id}/approve`、`/graph/entity/{id}`、`/graph/path`、`/stream/{task_id}`（SSE） |
| `knowledge/processor/review_process/schema/review_schema.py`（新） | 请求/响应模型（含证据链、风险等级、审批） |
| `knowledge/processor/review_process/services/review_service.py`（新） | 审查任务编排 + 审批恢复 |
| `knowledge/utils/auth_util.py`（新） | 三个服务共用的 CORS 白名单 + `X-API-Key` 鉴权（原生 ASGI 中间件，不破坏 SSE） |
| `knowledge/utils/file_util.py`（新） | 无三方依赖的上传文件名净化（可离线单测） |
| `knowledge/utils/task_util.py` | 节点中文名映射 |

### 3.8 前端

| 文件 | 实现要点 |
|---|---|
| `knowledge/processor/query_process/front/chat.html` | 电商合规问答；主体（商家/店铺）选择 |
| `knowledge/processor/review_process/front/review.html` | 审查提交 + 证据链/风险等级展示 + 人工审批按钮 + SSE 进度（审查类型为电商 9 类） |
| `knowledge/processor/import_process/front/import.html` | 上传前端（节点进度展示） |

### 3.9 示例数据 / 评测 / 测试

| 文件 | 实现要点 |
|---|---|
| `knowledge/data/sample_docs/` | 三份虚构示例：商品详情页 HTML / 商家资质档案 / 平台违规处罚决定书 |
| `knowledge/eval/qa_pairs.jsonl` | 55 条样本（违禁词 / 资质 / 违规整改 / 问题货源 / 类目价格 / 品牌授权 / 事实 / 商品综合；正负样本混合，日期用相对 token 防腐烂） |
| `knowledge/eval/evaluation.json` | 评测元信息与期望分布（按审查类型/意图/规则/风险等级） |
| `knowledge/test/compliance/test_commerce_domain.py` | 领域词表、属性、文档类型、合规维度、Prompt 区块 |
| `knowledge/test/compliance/test_compliance_rules.py` | 规则解析工具 + R1~R6 命中/不命中 + 风险等级汇总 |
| `knowledge/test/compliance/test_graph_query.py` | 跳数钳制 / Cypher 生成 / 结果整理 |
| `knowledge/test/compliance/test_extraction.py` | 抽取清洗（标签白名单 / 截断 / 关系归一 / 属性白名单） |
| `knowledge/test/compliance/test_review_flow.py` | HITL interrupt → approve → 报告生成；`report` 类型必触发 |
| `knowledge/test/compliance/test_review_api.py` | 5 个接口 + 任务状态 + 离线图谱兜底 + R1+R2 端到端 |
| `knowledge/test/compliance/test_eval_dataset.py` | 评测集规则命中与风险等级回归、meta 一致性 |
| `knowledge/test/compliance/test_hardening.py` | 工程加固回归（CORS/鉴权、文件名净化、增量重建、实体名守卫） |

### 3.10 工程与文档

| 文件 | 实现要点 |
|---|---|
| `pyproject.toml`（新） | pytest（testpaths / pythonpath）与 ruff（line-length / lint 规则集）配置 |
| `.pre-commit-config.yaml`（新） | 提交前钩子：ruff / ruff-format / 基础文件检查 |
| `knowledge/requirements-dev.txt`（新） | 开发依赖（pytest / ruff / pre-commit） |
| `README.md` | 重写：电商合规审查定位、领域模型、规则表、架构、运行步骤、示例 |
| `docs/REFACTOR_PLAN.md` | 本文件 |
| `docs/电商合规审查_业务需求与流程设计.md` | 业务需求、输入输出契约、流程设计与需求可追溯性矩阵 |

---

## 4. 目标 LangGraph 拓扑（review）

![审查流程目标拓扑](review_topology.pdf)

---

## 5. API 契约（新增）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/review` | 提交审查（company, review_type, question, session_id?）→ `{review_id, task_id, status}` |
| GET | `/review/{review_id}` | 审查结果：conclusion / risk_level / evidence_chain / findings / report / needs_human_review |
| POST | `/review/{review_id}/approve` | 人工审批（approved, comment, reviewer）→ 恢复图执行 → 生成报告 |
| GET | `/graph/entity/{entity_id}` | 实体及其一跳关系（含关系属性） |
| GET | `/graph/path?from_name=&to_name=&max_hops=` | 两实体间路径（多跳） |
| GET | `/stream/{task_id}` | SSE：ready / progress / delta / final（+ review 事件） |

---

## 6. 合规规则（内置，`rules/compliance_rules.py`）

| 编号 | 规则 | 触发条件 | 严重度 |
|---|---|---|---|
| R1 | 违禁词/极限词宣传 | `PROMOTES.banned_words` 命中数 ≥ `COMPLIANCE_BANNED_WORD_HIT` | 高 |
| R2 | 资质缺失或过期 | `HOLDS_CERT` 状态为缺失，或 `expire_date` < 今天 + `COMPLIANCE_CERT_EXPIRE_DAYS` | 中 |
| R3 | 违规未整改 | 近 `COMPLIANCE_PENALTY_YEARS` 年存在 `SANCTIONED_BY` 且无「已整改」留痕 | 高 |
| R4 | 问题货源 | `SUPPLIES` 的任一端出现在 `COMPLAINED_IN` 的主体集合中 | 中 |
| R5 | 类目错放与价格违规 | `BELONGS_TO.is_misplaced` 为真，或 `PROMOTES.discount_rate` ≥ `COMPLIANCE_FAKE_PRICE_RATIO` | 中 |
| R6 | 品牌授权链穿透 | `AUTHORIZED_BY`/`SHARED_BRAND` 构成的链深度 ≥ `COMPLIANCE_PENETRATION_DEPTH` | 低 |

框架约定：规则只读 `ReviewContext`，`evaluate()` 返回 `ComplianceFinding` 或 `None`，
`overall_risk_level()` 取最严重项；单条规则异常不影响其它规则。

---

## 7. 风险 / 待确认项

1. **`item_name` 字段命名**：保留字段名（仅改语义与展示）——避免波及 Milvus schema、Neo4j、Mongo、前端。**已按此决策执行**。
2. **HITL 持久化**：`interrupt()` 需 checkpointer。现用 LangGraph 内存版（`MemorySaver`，纯 Python、重启丢状态）；生产建议换持久化后端（SQLite / Mongo）。
3. **关系属性**：Neo4j MERGE 语句已支持关系属性（属 schema 破坏性变更，历史图谱需重建）。
4. **上游授权**：参考的上游项目未声明许可证，公开分发前需确认（见 `LICENSE` 的来源与致谢段）。

---

## 8. 实施顺序（落地步骤）

1. 领域模型 `commerce_domain.py` + 合规规则库
2. 导入侧：抽取 Prompt → kg_graph_node（标签/关系/属性）→ 文档类型与分块元数据
3. 查询侧：state → 检索节点场景化 → 图谱多跳
4. review_process：state/config/base → 8 个节点 → graph 编排 → HITL checkpointer
5. API：review_router + schema + service；收尾 task_util / 前端
6. 示例数据 + 评测 + 测试
7. README + 注释 + 联调

---

## 9. 工程加固与质量基线

| 项 | 现状 |
|---|---|
| 依赖与密钥 | 三方依赖延迟导入；MinIO 凭证、MCP 令牌、API Key 全部走环境变量，代码无硬编码口令 |
| CORS / 鉴权 | 统一 `auth_util.install_security(app)`；CORS 白名单可配且禁止 `*`；`API_KEY` 设置后全部业务接口需带 `X-API-Key` |
| 上传安全 | `sanitize_filename` 防路径遍历 + 扩展名白名单 |
| 数据安全 | 导入改为**按主体增量重建**，不再 `drop_collection` 清空整表 |
| 代码质量 | `pyproject.toml` 统一 pytest / ruff 配置；`.pre-commit-config.yaml` 提交前检查 |
| 测试 | `pytest knowledge/test/compliance -q` 离线全绿（无需 Milvus / Neo4j / LLM） |
