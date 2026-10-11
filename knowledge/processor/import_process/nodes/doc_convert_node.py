"""HTML / Word → Markdown 转换节点（业务文档格式支持）。

平台治理文档大量以网页（商品详情页、平台公告）与 Word（资质档案、处罚决定书）形式发布，
而下游分块、向量化、图谱抽取统一消费 Markdown，因此新增本节点做格式归一。

实现约束：
- 纯 Python 标准库实现（html.parser / zipfile / xml.etree），不引入新三方依赖；
- 保留标题层级与表格结构（业务文档中表格承载关键数据：商品价格、折扣率、资质有效期）；
- PDF 仍走 MinerU（pdf_to_md_node），Markdown 直读，本节点只处理 HTML 与 Word。
"""

import os
import re
import zipfile
import xml.etree.ElementTree as ET
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import List

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError


# ==================== HTML → Markdown ====================

class _HtmlToMarkdownParser(HTMLParser):
    """把 HTML 转为 Markdown（保留标题层级、列表、表格结构）。"""

    _HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
    _SKIP_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: List[str] = []
        self._skip_depth = 0
        self._heading_level = 0
        self._buffer: List[str] = []
        self._list_item = False
        # 表格状态
        self._in_table = False
        self._table_rows: List[List[str]] = []
        self._current_row: List[str] = []
        self._current_cell: List[str] = []
        self._in_cell = False

    # ---------- 工具 ----------
    def _flush_paragraph(self) -> None:
        text = "".join(self._buffer).strip()
        self._buffer = []
        if not text:
            return
        if self._heading_level:
            self.blocks.append("#" * self._heading_level + " " + text)
        elif self._list_item:
            self.blocks.append("- " + text)
        else:
            self.blocks.append(text)
        self._heading_level = 0
        self._list_item = False

    # ---------- HTMLParser 回调 ----------
    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self._HEADING_TAGS:
            self._flush_paragraph()
            self._heading_level = self._HEADING_TAGS[tag]
        elif tag in ("p", "div", "section", "article"):
            self._flush_paragraph()
        elif tag == "br":
            self._flush_paragraph()
        elif tag == "li":
            self._flush_paragraph()
            self._list_item = True
        elif tag == "table":
            self._flush_paragraph()
            self._in_table = True
            self._table_rows = []
        elif tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in self._HEADING_TAGS:
            self._flush_paragraph()
        elif tag in ("p", "div", "section", "article", "li"):
            self._flush_paragraph()
        elif tag in ("td", "th"):
            self._current_cell.append("")
            self._current_row.append("".join(self._current_cell).strip())
            self._current_cell = []
            self._in_cell = False
        elif tag == "tr":
            if self._current_row:
                self._table_rows.append(self._current_row)
            self._current_row = []
        elif tag == "table":
            self._render_table()
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data:
            return
        text = data.replace("\n", " ").strip()
        if not text:
            return
        if self._in_cell:
            self._current_cell.append(text + " ")
        else:
            self._buffer.append(text + " ")

    def close(self) -> None:  # noqa: A003
        super().close()
        if self._in_table:
            self._render_table()
            self._in_table = False
        self._flush_paragraph()

    # ---------- 表格渲染 ----------
    def _render_table(self) -> None:
        if not self._table_rows:
            return
        header = self._table_rows[0]
        lines = ["| " + " | ".join(header) + " |",
                 "| " + " | ".join(["---"] * len(header)) + " |"]
        for row in self._table_rows[1:]:
            # 补齐列数
            row = row + [""] * (len(header) - len(row))
            lines.append("| " + " | ".join(row[:len(header)]) + " |")
        self.blocks.append("\n".join(lines))
        self._table_rows = []


def html_to_markdown(html_text: str) -> str:
    """把 HTML 文本转换为 Markdown 文本。"""
    parser = _HtmlToMarkdownParser()
    parser.feed(html_text)
    parser.close()
    md = "\n\n".join(b for b in parser.blocks if b.strip())
    return re.sub(r"\n{3,}", "\n\n", md).strip()


# ==================== DOCX → Markdown ====================

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_paragraph_text(paragraph: ET.Element) -> str:
    """拼接一个 Word 段落内的所有文本。"""
    return "".join(node.text or "" for node in paragraph.iter(f"{_W_NS}t")).strip()


def _docx_heading_level(paragraph: ET.Element) -> int:
    """识别 Word 段落的标题级别（Heading1..6 / 标题1..6）。"""
    style = paragraph.find(f"{_W_NS}pPr/{_W_NS}pStyle")
    if style is None:
        return 0
    val = (style.get(f"{_W_NS}val") or "").lower()
    match = re.search(r"(?:heading|标题)\s*([1-6])", val)
    return int(match.group(1)) if match else 0


def _docx_table_to_markdown(table: ET.Element) -> str:
    """把 Word 表格转为 Markdown 表格。"""
    rows: List[List[str]] = []
    for tr in table.findall(f"{_W_NS}tr"):
        cells = []
        for tc in tr.findall(f"{_W_NS}tc"):
            text = "".join(node.text or "" for node in tc.iter(f"{_W_NS}t")).strip()
            cells.append(text)
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows[1:]:
        row = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(row[:len(header)]) + " |")
    return "\n".join(lines)


def docx_to_markdown(docx_path: str) -> str:
    """读取 .docx（本质是 zip + XML），转换为 Markdown。"""
    try:
        with zipfile.ZipFile(docx_path, "r") as zf:
            with zf.open("word/document.xml") as fp:
                tree = ET.parse(fp)
    except Exception as e:
        raise FileProcessingError(f"解析 Word 文档失败: {e}", "doc_convert_node")

    body = tree.getroot().find(f"{_W_NS}body")
    if body is None:
        return ""
    blocks: List[str] = []
    for child in body:
        if child.tag == f"{_W_NS}p":
            text = _docx_paragraph_text(child)
            if not text:
                continue
            level = _docx_heading_level(child)
            blocks.append(("#" * level + " " + text) if level else text)
        elif child.tag == f"{_W_NS}tbl":
            md_table = _docx_table_to_markdown(child)
            if md_table:
                blocks.append(md_table)
    return "\n\n".join(blocks).strip()


# ==================== 节点 ====================

class DocConvertNode(BaseNode):
    """HTML / Word → Markdown 节点。

    职责：
    1. 读取 state.import_file_path 指向的 .html/.htm/.docx/.doc 文件；
    2. 转换为 Markdown 并落盘到 file_dir/<file_title>.md；
    3. 回填 state.md_path / state.md_content，交由下游 md_img_node → 分块。
    """

    name = "doc_convert_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        import_file_path = state.get("import_file_path")
        file_dir = state.get("file_dir")
        file_title = state.get("file_title") or "document"

        if not import_file_path or not file_dir:
            raise ValidationError("转换所需文件路径缺失", self.name)

        path = Path(import_file_path)
        suffix = path.suffix.lower()

        # 1. 按格式选择转换器
        if suffix in (".html", ".htm"):
            html_text = self._read_text(import_file_path)
            md_content = html_to_markdown(html_text)
        elif suffix == ".docx":
            md_content = docx_to_markdown(import_file_path)
        elif suffix == ".doc":
            # 旧版二进制 .doc 无标准库解析方案，提示改用 .docx 或先另存
            raise FileProcessingError(
                "暂不支持旧版 .doc 二进制格式，请另存为 .docx 或 PDF 后重试", self.name
            )
        else:
            raise ValidationError(f"doc_convert_node 不支持的格式: {suffix}", self.name)

        if not md_content.strip():
            raise FileProcessingError("文档转换结果为空，请检查源文件内容", self.name)

        # 2. 落盘为 Markdown，供 md_img_node 读取
        md_path = os.path.join(file_dir, f"{file_title}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # 3. 回填状态
        state["md_path"] = md_path
        state["md_content"] = md_content
        # 转换后按 Markdown 流程继续（md_img_node 会再次读取并以自身为准）
        state["is_md_read_enabled"] = True

        self.logger.info(f"文档转换完成: {suffix} → Markdown（{len(md_content)} 字符）")
        return state

    @staticmethod
    def _read_text(path: str) -> str:
        """读取文本文件，兼容 UTF-8 / GBK 编码。"""
        for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
            try:
                with open(path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


if __name__ == "__main__":
    demo = "<h1>关于XX旗舰店商品宣传的处理公告</h1><p>该店铺折扣率80%。</p><table><tr><th>商品</th><th>折扣率</th></tr><tr><td>深海鱼油软胶囊</td><td>80%</td></tr></table>"
    print(html_to_markdown(demo))
