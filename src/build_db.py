"""
向量数据库构建与加载模块
对应原 rag_system_code.py 中 LocalRAGSystem 的：
    build_or_load_db / _build_new_db / _preload_retrieval_models / warmup_models
    _build_bm25_index / _build_keyword_index / _add_to_keyword_index / _remove_from_keyword_index
以 Mixin 形式提供给主系统类使用，方法实现与原文件保持一致。
"""

import os
import time
from array import array
from concurrent.futures import ThreadPoolExecutor

import jieba
# 导入向量数据库
from langchain_chroma import Chroma
from langchain_core.documents import Document as LangchainDocument
from rank_bm25 import BM25Okapi

import config
from config import CHROMA_DB_DIR, PRELOAD_MODELS, KEYWORD_MAX_DF_RATIO


class VectorDBMixin:
    """向量库构建/加载与检索索引构建能力（供 LocalRAGSystem 多继承组合）"""

    def build_or_load_db(self) -> None:
        """构建或加载向量数据库"""
        if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
            print(f"加载已有向量库: {CHROMA_DB_DIR}")
            try:
                self.vector_store = Chroma(
                    persist_directory=CHROMA_DB_DIR,
                    embedding_function=self.embeddings,
                    collection_name='knowledge_base',
                )
                count = self.vector_store._collection.count()
                print(f"向量库加载完成，共 {count} 个向量")

                # 恢复chunks列表
                all_docs_data = self.vector_store.get()
                if all_docs_data and 'documents' in all_docs_data:
                    self.chunks = []
                    for doc_id, text, metadata in zip(
                        all_docs_data.get('ids', []),
                        all_docs_data.get('documents', []),
                        all_docs_data.get('metadatas', []),
                    ):
                        doc = LangchainDocument(
                            page_content=text,
                            metadata=metadata or {}
                        )
                        self.chunks.append(doc)

                self._manifest.load()
                if self._manifest.get_total_documents() == 0 and self.chunks:
                    self._rebuild_manifest_from_chunks()
            except Exception as e:
                print(f"加载向量库失败: {e}，将重新构建")
                self._build_new_db()
        else:
            self._build_new_db()

        # 构建检索索引
        if self.chunks:
            self._build_bm25_index()
            self._build_keyword_index()

        # [OPT-15] 后台预加载 embedding + rerank（chat 模型在 warmup_models 中加载）
        if PRELOAD_MODELS:
            self._preload_retrieval_models()

    def _preload_retrieval_models(self):
        """后台预加载检索侧模型，不阻塞主流程"""
        def _load():
            try:
                self.embeddings.embed_query("warmup")
                if config.ENABLE_RERANK and not config.FAST_MODE:
                    self.rerank_model.predict([["warmup", "warmup"]], batch_size=1)
            except Exception:
                pass

        ThreadPoolExecutor(max_workers=1).submit(_load)

    def warmup_models(self, preload_chat: bool = True):
        """
        [OPT-15] 显式预热所有模型。
        首次加载 qwen2.5:7b 可能需要 60-180s，应放在启动阶段而非首次问答时。
        """
        print("正在预热模型（首次加载较慢，请稍候）...")
        t_total = time.time()

        t = time.time()
        try:
            self.embeddings.embed_query("warmup")
            print(f"  Embedding 就绪  {time.time() - t:.1f}s")
        except Exception as e:
            print(f"  Embedding 失败: {e}")

        if config.ENABLE_RERANK and not config.FAST_MODE:
            t = time.time()
            try:
                self.rerank_model.predict([["warmup", "warmup"]], batch_size=1)
                print(f"  Rerank 就绪      {time.time() - t:.1f}s")
            except Exception as e:
                print(f"  Rerank 失败: {e}")

        if preload_chat:
            t = time.time()
            try:
                self.chat_model.invoke("你好")
                print(f"  Chat LLM 就绪    {time.time() - t:.1f}s")
            except Exception as e:
                print(f"  Chat LLM 失败: {e}")

        print(f"模型预热完成，总耗时 {time.time() - t_total:.1f}s\n")

    def _build_new_db(self):
        """构建新的向量库"""
        print("首次运行，开始构建向量库...")
        docs = self._load_documents()
        if not docs:
            print("未找到文档，无法构建向量库")
            return

        self.chunks = self._split_documents(docs)
        if not self.chunks:
            print("文档切分后无有效内容")
            return

        for i, chunk in enumerate(self.chunks):
            chunk.metadata["chunk_index"] = i

        print(f"正在向量化 {len(self.chunks)} 个文本块...")
        try:
            self.embeddings.embed_query("warmup")
            self.vector_store = Chroma.from_documents(
                documents=self.chunks,
                ids=[chunk.metadata["chunk_id"] for chunk in self.chunks],
                persist_directory=CHROMA_DB_DIR,
                embedding=self.embeddings,
                collection_name='knowledge_base',
                collection_metadata={
                    "hnsw:space": "cosine",
                    "hnsw:search_ef": 200,
                    },
            )
            print(f"向量库构建完成，共 {len(self.chunks)} 个向量")
            self._rebuild_manifest_from_chunks()
        except Exception as e:
            print(f"向量库构建失败: {e}")
            raise

    # ==================== 索引构建 ====================
    def _build_bm25_index(self):
        """构建BM25索引"""
        tokenized_docs = []
        for d in self.chunks:
            tokens = [t for t in jieba.cut(d.page_content.strip())
                      if t.strip() and t not in self._stopwords]
            tokenized_docs.append(tokens)
        self._bm25_index = BM25Okapi(tokenized_docs)

    def _build_keyword_index(self):
        """
        构建倒排关键词索引。
        优化：高频词剪枝（出现在>30%文档中的词不建索引）+ array('I') 替代 set 压缩内存。
        """
        self._keyword_index: dict[str, array] = {}
        doc_count = len(self.chunks)
        if doc_count == 0:
            return

        # 第一遍：统计词频（document frequency）
        word_df: dict[str, int] = {}
        for idx, doc in enumerate(self.chunks):
            words = set(jieba.cut(doc.page_content.strip()))
            words = {w for w in words if w not in self._stopwords and len(w) > 1}
            for word in words:
                word_df[word] = word_df.get(word, 0) + 1

        # 第二遍：跳过高频词建索引，用 array('I') 存储 postings
        threshold = int(doc_count * KEYWORD_MAX_DF_RATIO)
        pruned_count = 0
        for idx, doc in enumerate(self.chunks):
            words = set(jieba.cut(doc.page_content.strip()))
            words = {w for w in words if w not in self._stopwords and len(w) > 1}
            for word in words:
                df = word_df.get(word, 0)
                if df > threshold:
                    pruned_count += 1
                    continue
                if word not in self._keyword_index:
                    self._keyword_index[word] = array('I')
                self._keyword_index[word].append(idx)

        print(f"倒排索引构建完成: {len(self._keyword_index)} 个词, 剪枝 {pruned_count} 个高频词posting")

    def _add_to_keyword_index(self, chunks: list, start_idx: int):
        """增量添加 chunk 到倒排索引（避免全量重建）"""
        if self._keyword_index is None:
            self._build_keyword_index()
            return
        doc_count = len(self.chunks)
        threshold = int(doc_count * KEYWORD_MAX_DF_RATIO) if doc_count > 0 else 0
        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            words = set(jieba.cut(chunk.page_content.strip()))
            words = {w for w in words if w not in self._stopwords and len(w) > 1}
            # 简易高频检测：已存在的 postings 长度 > threshold 也跳过
            for word in words:
                existing_len = len(self._keyword_index.get(word, []))
                if existing_len > threshold:
                    continue
                if word not in self._keyword_index:
                    self._keyword_index[word] = array('I')
                self._keyword_index[word].append(idx)

    def _remove_from_keyword_index(self, chunk_indices: set):
        """从倒排索引中增量移除指定 chunk 的 postings"""
        if self._keyword_index is None:
            return
        for word in list(self._keyword_index.keys()):
            postings = self._keyword_index[word]
            new_postings = array('I', (idx for idx in postings if idx not in chunk_indices))
            if len(new_postings) == 0:
                del self._keyword_index[word]
            else:
                self._keyword_index[word] = new_postings
