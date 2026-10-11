#内置库
import json
from pathlib import Path

from knowledge.processor.import_process.base import  BaseNode,setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.config import get_config
from knowledge.schema.commerce_domain import infer_doc_type
class EntryNode(BaseNode):
    """
    入口节点（导入流程第一位）
    作用: 对上传的文件类型做判断，并路由到对应的解析分支。

    电商合规场景适配：在原有 PDF / Markdown 基础上，新增 HTML / Word（.docx/.doc）
    两类常见格式，统一走 doc_convert_node 转 Markdown。
    """

    name = "entry"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        处理文件类型的检测
        Args:
            state:ImportGraphState

        Returns:ImportGraphState  该节点处理之后的节点状态

        """

        config = get_config()

        #1. 获取导入文件的路径以及文件所在的目录
        self.log_step("Step1","{获取文件路径}")
        import_file_path = state.get("import_file_path")
        file_dir = state.get("file_dir")

        #2.简单校验一下文件路径以及所在目录
        self.log_step("Step2","{检查文件路径}")
        if not file_dir or not import_file_path:
            raise ValidationError("文件目录或者文件路径不存在",self.name)

        #3.使用标准的Path对象操作文件逻辑
        path = Path(import_file_path)

        #4. 获取上传文件的后缀 path.suffix直接拿到文件的后缀名
        suffix = path.suffix.lower()#转成小写

        #4.1 校验是否为受支持的文档格式
        if config.supported_doc_extensions and suffix not in config.supported_doc_extensions:
            raise ValidationError(f"文件类型{suffix}不支持", self.name)

        #5. 判断文件后缀并打上路由标志
        if suffix == '.pdf':
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = import_file_path
        elif suffix in ('.md', '.markdown'):
            state['is_md_read_enabled'] = True
            state['md_path'] = import_file_path
        elif suffix in ('.html', '.htm', '.docx', '.doc'):
            # 平台治理公告/违规处罚决定书常以网页或 Word 形式发布
            state['is_doc_convert_enabled'] = True
        else:
            self.logger.debug(f"文件类型{suffix}不支持")
            raise ValidationError(f"文件类型{suffix}不支持")

        #6. 获取文件的标题名,使用path.stem
        file_title = path.stem
        state['file_title'] = file_title

        #7. 补全业务文档元数据（文档类型自动推断；发布日期/来源可由上传方覆写）
        if not state.get("doc_type"):
            state['doc_type'] = infer_doc_type(file_title)
        if not state.get("source"):
            state['source'] = config.default_source

        return state

