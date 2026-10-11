# -*- coding: utf-8 -*-
"""生成 README / REFACTOR_PLAN 中的流程图 PDF。

用法（在项目根目录）：
    python docs/generate_diagrams.py

依赖：matplotlib（纯 Python，无其它依赖）。
输出：docs/architecture.pdf / docs/import_flow.pdf / docs/review_flow.pdf / docs/review_topology.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 中文字体 ----------
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",    # 黑体
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
_FONT_PATH = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)
if _FONT_PATH:
    font_manager.fontManager.addfont(_FONT_PATH)
    _family = font_manager.FontProperties(fname=_FONT_PATH).get_name()
else:
    _family = "sans-serif"
plt.rcParams["font.family"] = _family
plt.rcParams["axes.unicode_minus"] = False

# ---------- 配色 ----------
C_API = "#dbeafe"; C_API_E = "#2563eb"
C_WF = "#dcfce7"; C_WF_E = "#16a34a"
C_TOOL = "#fef3c7"; C_TOOL_E = "#d97706"
C_STORE = "#ede9fe"; C_STORE_E = "#7c3aed"
C_NODE = "#f8fafc"; C_NODE_E = "#475569"
C_STAR = "#fee2e2"; C_STAR_E = "#dc2626"
C_DEC = "#fff7ed"; C_DEC_E = "#ea580c"

BOX_STYLE = "round,pad=0.02,rounding_size=0.03"


def box(ax, cx, cy, w, h, text, fc=C_NODE, ec=C_NODE_E, fontsize=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle=BOX_STYLE, fc=fc, ec=ec, lw=1.4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            weight=weight, color="#111827")


def arrow(ax, x1, y1, x2, y2, text="", rad=0.0, fontsize=8.5, color="#64748b"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=1.3,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, text, ha="center", va="center",
                fontsize=fontsize, color=color,
                bbox=dict(fc="white", ec="none", pad=1))


def line(ax, x1, y1, x2, y2, color="#64748b", lw=1.3):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle="round")


def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved:", path)


# ============================================================
# 1) 技术架构（分层）
# ============================================================
def architecture():
    fig, ax = new_fig(11, 7.2)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    def layer(y, h, title, fc, ec):
        ax.add_patch(FancyBboxPatch((0.4, y), 9.2, h, boxstyle=BOX_STYLE,
                                    fc=fc, ec=ec, lw=1.6))
        ax.text(0.75, y + h / 2, title, ha="left", va="center",
                fontsize=11, weight="bold", color=ec)

    # API 层
    layer(8.15, 1.55, "FastAPI\nAPI 层", C_API, C_API_E)
    for x, t in [(2.9, "导入 :8000\n/import"), (5.0, "查询 :8001\n/chat.html"), (7.1, "审查 :8002\n/review")]:
        box(ax, x, 8.85, 1.75, 0.85, t, "white", C_API_E, 9.5)

    # 工作流层
    layer(5.85, 1.55, "LangGraph\n工作流引擎", C_WF, C_WF_E)
    for x, t in [(2.9, "导入流程\n（9 节点）"), (5.0, "查询流程\n（9 节点）"), (7.1, "审查流程\n（8 节点）")]:
        box(ax, x, 6.55, 1.75, 0.85, t, "white", C_WF_E, 9.5)

    # 工具层
    layer(3.35, 2.05, "工具层", C_TOOL, C_TOOL_E)
    tools_row1 = [(2.0, "LLM 客户端"), (3.85, "BGE-M3 向量化"), (5.7, "BGE-Rerank"), (7.55, "合规规则引擎")]
    tools_row2 = [(2.0, "SSE 推送 / 任务追踪"), (3.85, "Mongo 历史"), (5.7, "路径 / 配置"), (7.55, "Milvus / Neo4j / MinIO 客户端")]
    for x, t in tools_row1:
        box(ax, x, 4.75, 1.7, 0.6, t, "white", C_TOOL_E, 8.8)
    for x, t in tools_row2:
        box(ax, x, 3.95, 1.7, 0.6, t, "white", C_TOOL_E, 8.8)

    # 存储层
    layer(1.0, 1.55, "存储层", C_STORE, C_STORE_E)
    for x, t in [(2.35, "Milvus\n向量"), (4.2, "Neo4j\n图谱"), (6.05, "MongoDB\n历史"), (7.9, "MinIO\n对象存储")]:
        box(ax, x, 1.7, 1.55, 0.85, t, "white", C_STORE_E, 9.5)

    # 层间箭头
    for x in (2.9, 5.0, 7.1):
        arrow(ax, x, 8.15, x, 7.4)
        arrow(ax, x, 5.85, x, 5.4)
    arrow(ax, 5.0, 3.35, 5.0, 2.55)

    save(fig, "architecture.pdf")


# ============================================================
# 2) 导入流程（9 节点）
# ============================================================
def import_flow():
    fig, ax = new_fig(11, 6.2)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    # 入口
    box(ax, 1.0, 5.0, 1.1, 0.9, "entry", C_NODE, C_NODE_E, 10.5, "bold")

    # 四条格式分支
    branch_y = {"pdf": 8.6, "html": 6.9, "md": 5.0, "other": 2.6}
    arrow(ax, 1.55, 5.25, 2.5, branch_y["pdf"], "PDF", rad=-0.25)
    arrow(ax, 1.55, 5.1, 2.5, branch_y["html"], "HTML / Word", rad=-0.18)
    line(ax, 1.55, 5.0, 2.5, 5.0)
    ax.text(2.0, 5.12, "Markdown", fontsize=8.5, color="#64748b", ha="center")
    arrow(ax, 1.55, 4.8, 2.5, branch_y["other"] + 0.45, "其它", rad=0.2)

    box(ax, 3.5, branch_y["pdf"], 1.7, 0.75, "pdf_to_md", "white", C_NODE_E, 9.5)
    box(ax, 3.5, branch_y["html"], 2.15, 0.75, "doc_convert\n（保留表格）", "white", C_NODE_E, 9)
    box(ax, 3.5, branch_y["other"] + 0.45, 1.2, 0.7, "END", C_NODE, C_NODE_E, 9.5)

    # 汇聚线 -> document_split
    conv_x, conv_y = 5.4, 5.0
    box(ax, conv_x, conv_y, 1.8, 0.75, "document_split", C_WF, C_WF_E, 9.5)
    line(ax, 4.35, branch_y["pdf"], conv_x, conv_y + 0.3)
    line(ax, 4.6, branch_y["html"], conv_x, conv_y + 0.2)
    line(ax, 4.4, 5.0, conv_x - 0.9, conv_y)

    # 主链
    nodes = ["item_name_\nrecognition", "knowledge_graph", "import_milvus", "bge_embedding_\nchunks"]
    xs = [7.3, 8.9, 7.3, 5.7]; ys = [5.0, 3.2, 1.7, 1.7]
    box(ax, xs[0], ys[0], 1.9, 0.85, nodes[0], C_WF, C_WF_E, 9)
    box(ax, xs[1], ys[1], 1.8, 0.8, nodes[1], C_WF, C_WF_E, 9.5)
    box(ax, xs[2], ys[2], 1.7, 0.75, nodes[2], C_WF, C_WF_E, 9.5)
    box(ax, xs[3], ys[3], 1.9, 0.8, nodes[3], C_WF, C_WF_E, 9)
    arrow(ax, conv_x + 0.9, conv_y, xs[0] - 0.95, ys[0])
    arrow(ax, xs[0] + 0.7, ys[0] - 0.5, xs[1] - 0.8, ys[1] + 0.4, rad=0.15)
    arrow(ax, xs[1] - 0.5, ys[1] - 0.45, xs[2] + 0.6, ys[2] + 0.35, rad=0.1)
    arrow(ax, xs[2] - 0.85, ys[2], xs[3] + 0.95, ys[3])

    # 标注
    ax.text(4.9, 3.2, "LLM 抽取实体/关系\n→ 写入 Neo4j", fontsize=8, color="#16a34a", ha="center", va="center")
    ax.text(6.5, 0.85, "双向量写入 Milvus", fontsize=8, color="#16a34a", ha="center")

    save(fig, "import_flow.pdf")


# ============================================================
# 3) 审查流程（8 节点 + HITL 分支）
# ============================================================
def review_flow():
    fig, ax = new_fig(10, 7.8)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    chain = [
        ("intent_recognize\n（意图识别）", 8.8),
        ("query_rewrite\n（改写 + HyDE）", 7.55),
        ("multi_search\n（向量 / 稀疏 / 图谱多跳）", 6.3),
        ("rrf_rerank\n（RRF 融合 + 重排）", 5.05),
        ("evidence_chain\n（证据链构建）", 3.8),
        ("compliance_check\n（合规规则检查）", 2.55),
    ]
    xs = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    for (label, y), x in zip(chain, xs):
        box(ax, x, y, 3.0, 0.95, label, C_WF, C_WF_E, 9.5)
    for i in range(len(chain) - 1):
        arrow(ax, 5.0, chain[i][1] - 0.48, 5.0, chain[i + 1][1] + 0.48)

    # 分支：合规检查 -> hitl / report_generate
    box(ax, 2.2, 1.15, 2.1, 0.9, "hitl\n（人工复核 interrupt）", C_STAR, C_STAR_E, 9)
    box(ax, 7.4, 1.15, 2.3, 0.9, "report_generate\n（审查报告）", C_WF, C_WF_E, 9.5)
    box(ax, 9.3, 2.9, 0.95, 0.6, "END", C_NODE, C_NODE_E, 9.5)

    arrow(ax, 4.2, 2.55 - 0.15, 3.0, 1.15 + 0.45, "高风险", rad=0.2, color="#dc2626")
    arrow(ax, 6.0, 2.55 - 0.4, 7.4, 1.15 + 0.45, "无高风险", rad=-0.15, color="#16a34a")
    arrow(ax, 2.2, 1.6, 6.3, 1.15, "approve 恢复", rad=-0.12, color="#dc2626")
    arrow(ax, 8.55, 1.15, 9.0, 2.6, rad=-0.2)

    ax.text(0.35, 0.35, "LangGraph interrupt() 暂停，等待 POST /review/{id}/approve",
            fontsize=8.5, color="#dc2626", ha="left")

    save(fig, "review_flow.pdf")


# ============================================================
# 4) 审查流程目标拓扑（REFACTOR_PLAN §4）
# ============================================================
def review_topology():
    fig, ax = new_fig(11.5, 7.6)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    box(ax, 5.0, 9.2, 2.6, 0.8, "intent_recognize\n（事实 / 路径 / 合规 / 报告）", C_WF, C_WF_E, 9.5)

    # 三分支
    box(ax, 1.7, 7.3, 2.2, 0.8, "graph_path_search\n（路径查询）", "white", C_NODE_E, 9)
    box(ax, 5.0, 7.3, 2.5, 0.8, "query_rewrite + HyDE\n（事实 / 合规查询）", "white", C_NODE_E, 9)
    box(ax, 8.2, 7.3, 2.0, 0.8, "直接走检索底座\n（报告生成）", "white", C_NODE_E, 9)
    arrow(ax, 4.0, 9.2, 1.7, 7.7, rad=0.2)
    arrow(ax, 5.0, 8.8, 5.0, 7.7)
    arrow(ax, 6.0, 9.2, 8.2, 7.7, rad=-0.2)

    # 中路继续
    box(ax, 5.0, 5.7, 3.2, 0.9, "multi_search（并行）\nvector │ sparse/BM25 │ kg_multi_hop", C_WF, C_WF_E, 8.8)
    box(ax, 5.0, 4.4, 2.6, 0.7, "rrf → rerank", C_WF, C_WF_E, 9.5)
    arrow(ax, 5.0, 6.9, 5.0, 6.15)
    arrow(ax, 5.0, 5.25, 5.0, 4.75)

    # 左路汇入
    line(ax, 1.7, 6.9, 1.7, 3.55)
    arrow(ax, 1.7, 3.55, 3.4, 3.3, rad=0.1)

    # evidence_chain / compliance_check
    box(ax, 5.0, 3.3, 3.2, 0.8, "evidence_chain\n（图谱路径 + 文档片段 → 证据链）", C_WF, C_WF_E, 8.8)
    box(ax, 5.0, 2.1, 3.6, 0.8, "compliance_check\n（规则命中 → findings / risk_level）", C_WF, C_WF_E, 8.8)
    arrow(ax, 5.0, 4.05, 5.0, 3.7)
    arrow(ax, 5.0, 2.9, 5.0, 2.5)

    # 风险分支（是/否 直接标注在箭头上，避免判断框与节点重叠）
    box(ax, 2.4, 0.9, 1.9, 0.8, "hitl\n（interrupt）", C_STAR, C_STAR_E, 9.5)
    box(ax, 7.6, 0.9, 2.3, 0.8, "report_generate → END", C_WF, C_WF_E, 9)
    arrow(ax, 4.0, 1.85, 2.7, 1.35, "是（高风险）", rad=0.25, color="#dc2626")
    arrow(ax, 6.0, 1.85, 7.6, 1.35, "否", rad=-0.25, color="#16a34a")
    arrow(ax, 4.3, 0.9, 6.45, 0.9, "approve 恢复", color="#dc2626")

    save(fig, "review_topology.pdf")


if __name__ == "__main__":
    architecture()
    import_flow()
    review_flow()
    review_topology()
    print("全部图形生成完成")
