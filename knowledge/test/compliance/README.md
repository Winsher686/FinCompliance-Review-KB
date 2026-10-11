# 电商合规测试套件

覆盖领域模型、抽取清洗、图谱查询、合规规则、审查流程（HITL）、API、
LLM 客户端加固、可观测性、检索并发调度与评测集/运行器的离线测试。

## 运行

```bash
# 全部（pytest，137 passed / 8 skipped）
pytest knowledge/test/compliance -q

# 或逐个脚本直接运行（会自动把项目根加入 sys.path）
python knowledge/test/compliance/test_commerce_domain.py
python knowledge/test/compliance/test_compliance_rules.py
python knowledge/test/compliance/test_graph_query.py
python knowledge/test/compliance/test_concurrency.py
python knowledge/test/compliance/test_llm_client.py
python knowledge/test/compliance/test_observability.py
python knowledge/test/compliance/test_eval_dataset.py
python knowledge/test/compliance/test_eval_runner.py
python knowledge/test/compliance/test_devops_config.py
python knowledge/test/compliance/test_review_flow.py
python knowledge/test/compliance/test_review_api.py
```

## 用例清单

| 文件 | 校验内容 | 依赖 |
| --- | --- | --- |
| `test_commerce_domain.py` | 电商实体/关系白名单、关系属性、文档类型推断、合规维度、Prompt 区块 | 无 |
| `test_compliance_rules.py` | 金额/比例/日期解析、R1~R6 命中/不命中、风险等级汇总 | 无 |
| `test_graph_query.py` | 跳数钳制、Cypher 生成、结果整理、**UNWIND 批量查询语句与分组** | 无（Neo4j 驱动可选） |
| `test_concurrency.py` | 三路检索并发齐发、墙钟 ≈ max 而非 sum、分支超时/异常隔离、分支耗时可观测、图谱批量下推 | 无（全用桩分支） |
| `test_llm_client.py` | JSON 围栏/括号抽取、瞬时错误重试退避、非瞬时快速失败、str prompt 不被拆成字符 | 无（打桩） |
| `test_observability.py` | trace_id 派生、瀑布渲染、span 摘要、指标分位数、关闭后空操作、端到端产出 trace | langgraph |
| `test_eval_dataset.py` | 评测集规模/结构/正负样本覆盖、确定性指标满分、元信息不漂移、**相对日期不腐烂**、HITL 口径 | 无 |
| `test_eval_runner.py` | 无 LLM 时 Judge 优雅跳过、评分聚合与垃圾输出降级、报告送评、离线 main() 退出码 | 无（打桩） |
| `test_devops_config.py` | Dockerfile（多阶段/非 root/CUDA 对齐）、compose 服务与健康检查、GPU 叠加、CI job 与最小依赖集 | pyyaml |
| `test_extraction.py` | 非法标签过滤、超长名截断、关系归一、属性白名单 | pymilvus / langchain（缺失自动跳过） |
| `test_review_flow.py` | 高风险 interrupt → 审批恢复 → 报告生成 | langgraph |
| `test_review_api.py` | 7 个接口（含 `/trace`、`/metrics`）+ 任务状态 + 离线图谱兜底 | fastapi / uvicorn / httpx |
| `test_hardening.py` | 加固项回归（CORS/鉴权、文件名净化、Milvus 增量重建、实体名守卫） | 部分需 pymilvus（自动跳过） |

## 合规规则（内置 R1~R6）

| 规则 | 名称 | 触发维度 |
| --- | --- | --- |
| R1 | 违禁词/极限词宣传 | 商品宣传命中违禁词 / 极限词（商品宣传用语） |
| R2 | 资质缺失或过期 | 受管控类目所需资质缺失或已过期 |
| R3 | 违规未整改 | 近 N 年存在违规处罚但未见整改 |
| R4 | 问题货源 | 上游供应商为被投诉主体 |
| R5 | 类目错放与价格违规 | 类目错放，或促销折扣率异常（涉嫌虚构原价） |
| R6 | 品牌授权链穿透 | 存在多层级品牌授权链（信息提示，低风险） |

阈值通过环境变量配置：`COMPLIANCE_BANNED_WORD_HIT`、`COMPLIANCE_CERT_EXPIRE_DAYS`、
`COMPLIANCE_PENALTY_YEARS`、`COMPLIANCE_FAKE_PRICE_RATIO`、`COMPLIANCE_PENETRATION_DEPTH`
（见 `knowledge/.env.example`）。

## 离线可跑说明

- 向量库（Milvus）、图库（Neo4j）、LLM 均非必需：检索/改写/生成节点在依赖不可用时走模板兜底；
  图谱工具在无驱动时返回空结果而非报错。
- `test_extraction.py` 因导入侧依赖 `pymilvus` / `langchain` 等重型三方库，
  在精简环境下会自动 **skip**（不报错）；在完整环境（`knowledge/.venv`）中会完整执行。

## 安装测试依赖

```bash
pip install -r requirements-dev.txt
```

或在精简环境中单独安装：

```bash
pip install pytest fastapi uvicorn httpx
```
