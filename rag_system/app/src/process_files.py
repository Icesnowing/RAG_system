"""
文档加载、文本处理与切分模块
对应原 rag_system_code.py 中 LocalRAGSystem 的：
    _load_docx_with_heading / _load_pdf_with_fallback / _load_documents / _split_documents
以 Mixin 形式提供给主系统类使用，方法实现与原文件保持一致。
"""

import os
import re
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入文档加载器
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
# 导入文本切分器
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from docx import Document
from langchain_core.documents import Document as LangchainDocument

from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, MAX_WORKERS


class DocumentProcessMixin:
    """文档加载、文本处理与切分能力（供 LocalRAGSystem 多继承组合）"""

    def _load_docx_with_heading(self, docx_path: str) -> LangchainDocument:
        """
        [OPT-13] 增强版Word文档加载：兼容.doc/.docx，更健壮的标题检测
        原理：原版仅通过style name正则匹配，对自定义样式文档无法识别标题。
              增加对内置Heading样式ID的检测，覆盖更多文档类型。
        """
        try:
            word_doc = Document(docx_path)
        except Exception:
            print("=================")
            return LangchainDocument(
                page_content=f"[加载失败] {docx_path}",
                metadata={"source": docx_path, "load_error": True}
            )

        markdown_lines = []
        for para in word_doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            style_name = (para.style.name or "").lower() if para.style else ""
            level = 0

            # 方法1：样式名称正则匹配 (标题 1, Heading 1, head 1)
            match = re.search(r'(?:head(?:ing)?|标题)\s*(\d+)', style_name, re.IGNORECASE)
            if match:
                level = int(match.group(1))
            else:
                # [OPT-13] 方法2：通过样式ID判断内置标题
                style_id = (para.style.style_id or "").lower() if para.style else ""
                match_id = re.search(r'heading(\d+)', style_id)
                if match_id:
                    level = int(match_id.group(1))

            if 1 <= level <= 3:
                markdown_lines.append(f"{'#' * level} {text}")
            else:
                markdown_lines.append(text)

        full_text = "\n\n".join(markdown_lines)
        return LangchainDocument(
            page_content=full_text,
            metadata={"source": docx_path, "file_type": "docx"}
        )

    def _load_pdf_with_fallback(self, pdf_path: str) -> List[LangchainDocument]:
        """
        [OPT-05] 带容错的PDF加载：先尝试完整加载，失败则逐页加载。
        原理：PyPDFLoader的load()一次性加载所有页，遇到损坏页会整体失败；
              逐页加载可以跳过损坏页，保留有效页的数据。
        """
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            if docs:
                for i, doc in enumerate(docs):
                    doc.metadata["page"] = i + 1
                    doc.metadata["source"] = str(pdf_path)
                    doc.metadata["file_type"] = "pdf"
                return docs
        except Exception:
            pass

        # 逐页加载容错
        docs = []
        try:
            loader = PyPDFLoader(str(pdf_path))
            for page_num in range(1000):  # 安全上限
                try:
                    page_docs = loader.load()
                    for doc in page_docs:
                        if doc.metadata.get("page", -1) == page_num:
                            docs.append(doc)
                            break
                except Exception:
                    break
        except Exception:
            pass

        if not docs:
            docs = [LangchainDocument(
                page_content=f"[PDF加载失败] {os.path.basename(pdf_path)}",
                metadata={"source": str(pdf_path), "load_error": True, "file_type": "pdf"}
            )]

        return docs

    def _load_documents(self) -> List[LangchainDocument]:
        """
        [OPT-06] 并行加载目录下所有文档（txt/pdf/docx/doc）
        原理：文档加载是I/O密集型操作，并行加载可大幅减少启动时间。
        """
        docs: List[LangchainDocument] = []
        docs_path = Path(DOCS_DIR)

        if not docs_path.exists():
            docs_path.mkdir(parents=True)
            print(f"已创建文档目录: {DOCS_DIR}，请放入文档后重新运行")
            return []

        # 收集所有文件
        txt_files = list(docs_path.glob('*.txt'))
        pdf_files = list(docs_path.glob('*.pdf'))
        docx_files = list(docs_path.glob('*.docx'))
        doc_files = list(docs_path.glob('*.doc'))
        all_files = txt_files + pdf_files + docx_files + doc_files

        if not all_files:
            print(f"目录 {DOCS_DIR} 下未找到文档")
            return []

        print(f"发现 {len(all_files)} 个文档文件 (txt:{len(txt_files)} pdf:{len(pdf_files)} docx/doc:{len(docx_files)+len(doc_files)})")

        # [OPT-06] 并行加载
        def load_single_file(file_path: Path):
            ext = file_path.suffix.lower()
            try:
                if ext == '.txt':
                    loader = TextLoader(str(file_path), encoding='utf-8')
                    loaded = loader.load()
                    return loaded

                elif ext == '.pdf':
                    return self._load_pdf_with_fallback(str(file_path))

                elif ext in ('.docx', '.doc'):
                    doc = self._load_docx_with_heading(str(file_path))
                    return [doc]

            except Exception as e:
                return [LangchainDocument(
                    page_content=f"[加载失败] {file_path.name}: {str(e)}",
                    metadata={"source": str(file_path), "load_error": True}
                )]

        results = []
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(all_files))) as executor:
            futures = {executor.submit(load_single_file, f): f for f in all_files}
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    loaded = future.result(timeout=60)
                    if loaded:
                        results.extend(loaded)
                except Exception as e:
                    pass

        docs = results
        print(f"文档加载完成，共 {len(docs)} 个文档对象")
        return docs

    def _split_documents(self, docs: List[LangchainDocument]) -> List[LangchainDocument]:
        """
        [OPT-01] 关键修复：对所有文档类型执行文本切分。

        原版bug：if/elif逻辑导致只有.docx被切分，.txt和.pdf加载后直接丢弃。
        修复：统一使用RecursiveCharacterTextSplitter对所有文档切分，
              .docx额外经过MarkdownHeaderTextSplitter保留章节结构。

        原理：文本切分是向量化和检索的基础，未经切分的文档无法进入向量库。
        """
        final_chunks: List[LangchainDocument] = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=['\n\n', '\n', '。 ', '. ', '！', '？', '!', '?', ' ', ''],
        )

        for doc in docs:
            file_name = os.path.basename(doc.metadata.get('source', ''))
            content = doc.page_content or ''
            if len(content.strip()) < 5:
                continue

            # [OPT-01] .docx 额外经过标题分层切分
            if file_name.endswith(('.docx', '.doc')):
                try:
                    headers_to_split_on = [
                        ("#", "H1"),
                        ("##", "H2"),
                        ("###", "H3"),
                    ]
                    md_splitter = MarkdownHeaderTextSplitter(
                        headers_to_split_on=headers_to_split_on
                    )
                    header_splits = md_splitter.split_text(content)

                    chunks = splitter.split_documents(header_splits)
                    print(f"已切分{file_name}，共 {len(chunks)} 个文本块")
                except Exception:
                    chunks = splitter.split_documents([doc])
            else:
                # [OPT-01] txt/pdf 统一切分
                chunks = splitter.split_documents([doc])

            # 补充源文件元数据
            for chunk in chunks:
                if "source" not in chunk.metadata:
                    chunk.metadata["source"] = doc.metadata.get("source", file_name)
                if "file_type" not in chunk.metadata:
                    chunk.metadata["file_type"] = doc.metadata.get("file_type", "unknown")

            final_chunks.extend(chunks)

        print(f"文本切分完成，共 {len(final_chunks)} 个文本块")
        return final_chunks
