# 电商合规示例文档

本目录提供用于验证「导入 → 图谱 → 审查」全链路的示例文档（虚构数据，仅作演示）。

| 文件 | 文档类型 | 覆盖能力 |
| --- | --- | --- |
| `XX旗舰店_商品详情页.html` | 商品详情页 | HTML → Markdown 转换（保留 `<table>`）、商品/类目/价格、极限词宣传、资质要求 |
| `广州XX贸易有限公司_商家资质档案.md` | 商家资质档案 | 商家主体、资质证照缺失、品牌授权、货源供应商、经营指标（GMV/在售数）、Markdown 表格 |
| `平台治理中心_违规处罚决定书.md` | 违规处罚决定 | 平台治理机构 / 处罚金额 / 处理单号 / 违规事由与整改状态、章节条款结构 |

三份文档共同覆盖了合规规则 R1（违禁词/极限词宣传）、R2（资质缺失或过期）、
R3（违规未整改）、R4（问题货源）、R5（类目错放与价格违规）、R6（品牌授权链穿透）的取证场景。

## 如何导入

1. 启动导入服务：`start_import_8000.bat`（依赖 `knowledge/.venv` 完整环境）；
2. 打开 `http://127.0.0.1:8000/`，上传上述文档；
3. 系统将自动：文档格式转换 → 按章节/条款分块（注入 doc_type / publish_date / source 等元数据）
   → BGE-M3 向量化写入 Milvus → LLM 抽取电商实体与关系写入 Neo4j。

## 如何离线验证（无需 Milvus / Neo4j / LLM）

示例数据的关系与经营指标已抽取为结构化 JSON，直接放在
`knowledge/eval/qa_pairs.jsonl` 中，可由规则引擎与审查图离线复算：

```bash
python knowledge/test/compliance/test_eval_dataset.py
python knowledge/test/compliance/test_review_flow.py
```

> 说明：`item_name` 字段在本项目中的语义为「主体名（商家/店铺名）」，
> 作为知识库的隔离与过滤维度。
