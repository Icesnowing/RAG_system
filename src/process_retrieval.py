"""
检索流程模块
对应原 rag_system_code.py 中 LocalRAGSystem 的：
    三路召回（关键词/BM25/向量）、RRF融合、CrossEncoder重排、chunk去重、
    单Query/多Query检索、token预算截断、检索结果文档压缩
以 Mixin 形式提供给主系统类使用，方法实现与原文件保持一致。
"""

import re
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import jieba
import numpy as np
#导入对话提示词模板
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document as LangchainDocument

import config
from config import (
    TOP_K,
    SIMILARITY_THRESHOLD,
    ENABLE_KEYWORD_AUGMENT,
    ENABLE_PARALLEL_RECALL,
    ENABLE_MULTI_QUERY,
    MULTI_QUERY_NUM,
    RERANK_CANDIDATE_LIMIT,
    RERANK_BATCH_SIZE,
    RERANK_TOP_N,
    ENABLE_DEDUP,
    DEDUP_METHOD,
    TEXT_DEDUP_THRESHOLD,
    DEDUP_THRESHOLD,
    MAX_CONTEXT_TOKENS,
    ENABLE_DOC_COMPRESSION,
    COMPRESSION_TARGET_LENGTH,
    COMPRESSION_RATIO,
    MAX_WORKERS,
)


class RetrievalMixin:
    """多路召回、融合重排与检索结果后处理能力（供 LocalRAGSystem 多继承组合）"""

    #================= 三路召回 ==================
    #================= 关键词匹配召回 ==================
    def _keyword_recall_with_augment(self, query: str, augmented_keywords: list = None) -> list:
        """
        增强版关键词召回：使用原始查询 + 增强关键词
        """
        # 如果有增强关键词，合并到查询中
        if augmented_keywords and ENABLE_KEYWORD_AUGMENT:
            # 将关键词拼接到查询中，增加权重
            enhanced_query = query + " " + " ".join(augmented_keywords)
            print(f"🔍 关键词召回增强查询: {enhanced_query}")
        else:
            enhanced_query = query

        tokenized_query = list(jieba.cut(enhanced_query.strip()))
        query_words = set(tokenized_query)

        # 过滤停用词
        query_words = {w for w in query_words if w not in self._stopwords}

        # 使用倒排索引快速计算
        doc_scores = {}
        for word in query_words:
            if word in self._keyword_index:
                for doc_idx in self._keyword_index[word]:
                    doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + 1

        scored = [(idx, score) for idx, score in doc_scores.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:TOP_K]

    def _keyword_recall(self, query: str, topk=TOP_K):
        tokenized_query = list(jieba.cut(query.strip())) #对查询语句做分词，拆成关键词列表
        query_words = set(tokenized_query) # 对关键词列表去重
        query_words = {w for w in query_words if w not in self._stopwords and len(w) > 1}

        if not query_words:
            return []
        # 使用倒排索引快速计算
        doc_scores = {}
        for word in query_words:
            if word in self._keyword_index:
                for doc_idx in self._keyword_index[word]:
                    doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + 1

        scored = [(idx, score) for idx, score in doc_scores.items()]
        scored.sort(key=lambda x: x[1], reverse=True) # 按匹配的关键词数量排序，从大到小
        return scored[:topk]

    #================= BM25召回 ==================
    def _bm25_recall_with_augment(self, query: str, augmented_keywords: list = None) -> list:
        """
        增强版BM25召回：使用原始查询 + 增强关键词
        """
        if augmented_keywords and ENABLE_KEYWORD_AUGMENT:
            # 将关键词拼接到查询中，增强BM25匹配
            enhanced_query = query + " " + " ".join(augmented_keywords)
            print(f"🔍 BM25召回增强查询: {enhanced_query}")
        else:
            enhanced_query = query

        tokenized_query = list(jieba.cut(enhanced_query.strip()))
        tokenized_query = [t for t in tokenized_query if t not in self._stopwords]
        if not tokenized_query:
            return []

        scores = self._bm25_index.get_scores(tokenized_query)
        ranked = np.argsort(scores)[::-1][:TOP_K]
        return [(int(i), float(scores[i])) for i in ranked if scores[i] > 0]

    def _bm25_recall(self, query: str, topk=TOP_K):

        # 对查询做分词
        tokenized_query = list(jieba.cut(query.strip()))
        tokenized_query = [t for t in tokenized_query if t not in self._stopwords]
        if not tokenized_query:
            return []
        scores = self._bm25_index.get_scores(tokenized_query)
        if len(scores) <= topk:
            ranked = np.argsort(scores)[::-1][:topk]
        else:
            top_indices = np.argpartition(scores, -topk)[-topk:]
            ranked = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(int(i), float(scores[i])) for i in ranked]

    #================= 向量检索召回 ==================
    def _vector_recall(self, query: str, topk=TOP_K):
        # [OPT-14] 直接调用向量库检索，避免每次创建retriever对象
        try:
            docs_scores = self.vector_store.similarity_search_with_relevance_scores(
                query, k=topk
            )
        except Exception:
            ret = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": topk, "score_threshold": SIMILARITY_THRESHOLD},
            )
            docs_scores = [(doc, doc.metadata.get("score", 0.0)) for doc in ret.invoke(query)]

        result = []
        for doc, score in docs_scores:
            if score < SIMILARITY_THRESHOLD:
                continue
            idx = doc.metadata.get("chunk_index")
            if idx is not None and 0 <= idx < len(self.chunks):
                result.append((int(idx), float(score)))
        return result

    #================== RRF融合 ==================
    def _rrf_fuse(self, lists, k=60):
        scores = {} # 初始化一个空字典，用于存储每个文档块的RRF分数，key是文档块索引，value是RRF分数
        #print(f"融合前候选：{len(lists)} 条\n")
        for lst in lists:
            #遍历列表里的每一条记录，enumerate(lst)给每个元素带上索引
            # rank：从0开始的排名，第一名是0，第二名是1
            # idx：列表里文档块索引
            # _：忽略，因为这里只需要索引
            for rank, (idx, _) in enumerate(lst):
                s = 1.0 / (rank + k) # RRF核心公式，k是平滑参数，防止除0错误
                # 把当前item的RRF分数累加到总字典里，如果idx存在字典里，分数就累加，否则默认是0。
                # 这里已经实现了去重，因为idx是文档的唯一ID，scores字典的key是唯一的，idx是字典里的key值
                scores[idx] = scores.get(idx, 0) + s
        return sorted(scores.items(), key=lambda x: x[1], reverse=True) # 按RRF分数从高到低排序

    #================== 重排 ==================
    def _cross_encoder_rerank(self, query: str, fused_indices: list):
        # [OPT-14] 限制送入重排模型的候选数量，显著降低CrossEncoder耗时
        limited = fused_indices[:RERANK_CANDIDATE_LIMIT]
        candidates = [self.chunks[idx].page_content for idx, _ in limited]
        docs_obj = [self.chunks[idx] for idx, _ in limited]
        pairs = [[query, c] for c in candidates]
        scores = self.rerank_model.predict(pairs, batch_size=RERANK_BATCH_SIZE).tolist()
        reranked = sorted(zip(docs_obj, scores), key=lambda x: x[1], reverse=True)
        return reranked[:RERANK_TOP_N]

    @staticmethod
    def _char_bigram_similarity(text_a: str, text_b: str) -> float:
        """基于字符二元组的Jaccard相似度，用于快速去重"""
        def bigrams(text: str) -> set:
            snippet = text[:300]
            return {snippet[i:i + 2] for i in range(len(snippet) - 1)}

        bg_a, bg_b = bigrams(text_a), bigrams(text_b)
        if not bg_a or not bg_b:
            return 0.0
        union = bg_a | bg_b
        return len(bg_a & bg_b) / len(union) if union else 0.0

    # ==================== [OPT-07] Chunk去重 ====================
    def _deduplicate_chunks(self, docs: List[LangchainDocument]) -> List[LangchainDocument]:
        """
        对重排后的chunk列表去重。
        [OPT-14] 默认使用字符相似度快速去重，避免每个chunk都调用embedding API。
        """
        if not ENABLE_DEDUP or len(docs) <= 1:
            return docs

        if DEDUP_METHOD == "text":
            deduped = []
            for doc in docs:
                content = doc.page_content[:500]
                if any(
                    self._char_bigram_similarity(content, kept.page_content[:500]) > TEXT_DEDUP_THRESHOLD
                    for kept in deduped
                ):
                    continue
                deduped.append(doc)
            return deduped

        deduped = []
        seen_embeddings = []
        for doc in docs:
            is_dup = False
            try:
                doc_vec = np.array(self.embeddings.embed_query(doc.page_content[:500]))
            except Exception:
                doc_vec = None

            if doc_vec is not None:
                for seen_vec in seen_embeddings:
                    sim = float(np.dot(doc_vec, seen_vec) /
                                (np.linalg.norm(doc_vec) * np.linalg.norm(seen_vec) + 1e-8))
                    if sim > DEDUP_THRESHOLD:
                        is_dup = True
                        break
                if not is_dup:
                    seen_embeddings.append(doc_vec)

            if not is_dup:
                deduped.append(doc)

        return deduped

    # ====================== 单Query检索 ======================
    def _single_query_retrieve(self, query: str) -> list:
        """单Query的三路召回"""
        augmented_keywords = self._augment_query(query)

        # [OPT-14] 三路召回并行执行（BM25/向量/关键词互不依赖）
        if ENABLE_PARALLEL_RECALL:
            with ThreadPoolExecutor(max_workers=3) as executor:
                f_bm25 = executor.submit(self._bm25_recall_with_augment, query, augmented_keywords)
                f_vec = executor.submit(self._vector_recall, query)
                f_kw = executor.submit(self._keyword_recall_with_augment, query, augmented_keywords)
                return [f_bm25.result(), f_vec.result(), f_kw.result()]

        bm25 = self._bm25_recall_with_augment(query, augmented_keywords)
        vec = self._vector_recall(query)
        kw = self._keyword_recall_with_augment(query, augmented_keywords)
        return [bm25, vec, kw]

    # ====================== 多Query检索======================
    #=====================生成多个query=====================
    def _generate_multi_queries(self, original_query: str) -> list[str]:
        """
        基于原始查询生成多个不同角度的查询语句（多Query扩展）
        :param original_query: 用户原始问题
        :return: 包含原始查询的多个查询列表
        """
        if not ENABLE_MULTI_QUERY:
            return [original_query]

        # 多Query生成提示词模板
        multi_query_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是查询扩展助手。根据用户问题生成{num}个不同角度的查询语句。
规则：
1. 保持核心语义，从不同表述/角度/关键词组合生成
2. 每个查询简洁，20字以内
3. 只输出查询语句，每行一个
4. 必须生成{num}个查询"""),
            ("human", "原始问题：{query}")
        ])
        try:
            chain = multi_query_prompt | self.multi_query_model
            response = chain.invoke({"query": original_query, "num": MULTI_QUERY_NUM})
            content = response.content if hasattr(response, 'content') else str(response)
            queries = [q.strip() for q in content.split('\n') if q.strip()]
            all_queries = list({original_query} | set(queries))[:MULTI_QUERY_NUM + 1]
            return all_queries
        except Exception:
            return [original_query]

    def _multi_query_retrieve(self, clean_query: str) -> list[LangchainDocument]:
        """
        多Query检索主流程：
        1. 生成多个扩展Query
        2. 每个Query独立执行三路召回
        3. 融合所有召回结果
        4. 重排得到最终结果
        """
        # 步骤1：生成多Query
        all_queries = self._generate_multi_queries(clean_query)

        # 步骤2：每个Query独立检索（多Query之间也并行）
        all_retrieval_results = []
        with ThreadPoolExecutor(max_workers=min(len(all_queries), MAX_WORKERS)) as executor:
            futures = [executor.submit(self._single_query_retrieve, q) for q in all_queries]
            for future in as_completed(futures):
                try:
                    all_retrieval_results.extend(future.result())
                except Exception:
                    pass

        # 步骤3：融合所有检索结果
        print(f"\n==== 融合所有Query的检索结果 ====")
        fused = self._rrf_fuse(all_retrieval_results)
        print(f"多Query融合后候选：{len(fused)} 条")

        # 步骤4：CrossEncoder重排（或 FAST_MODE 下直接取 RRF Top-N）
        reranked_docs = self._select_top_docs(clean_query, fused)
        return reranked_docs

    def _select_top_docs(self, query: str, fused: list) -> List[LangchainDocument]:
        """从重排或 RRF 结果中选取 Top-N 文档"""
        if not fused:
            return []
        if config.FAST_MODE or not config.ENABLE_RERANK:
            return [self.chunks[idx] for idx, _ in fused[:RERANK_TOP_N]]
        reranked = self._cross_encoder_rerank(query, fused)
        return [doc for doc, _ in reranked]

    def _retrieve_with_rerank(self, query: str) -> list[LangchainDocument]:
        """
        核心检索：预处理 → 三路召回 → RRF融合 → 重排/Top-N → 去重 → 截断
        """
        clean_query = self._preprocess_query(query)
        if not clean_query:
            return []

        if ENABLE_MULTI_QUERY:
            docs = self._multi_query_retrieve(clean_query)
        else:
            t0 = time.time()
            single_result = self._single_query_retrieve(clean_query)
            fused = self._rrf_fuse(single_result)
            if not fused:
                return []
            docs = self._select_top_docs(clean_query, fused)
            if config.ENABLE_PERF_LOG:
                print(f"  检索+融合+重排: {time.time() - t0:.2f}s")

        t1 = time.time()
        docs = self._deduplicate_chunks(docs)
        docs = self._truncate_by_token_budget(docs)
        if config.ENABLE_PERF_LOG:
            print(f"  去重+截断: {time.time() - t1:.2f}s")

        if ENABLE_DOC_COMPRESSION:
            docs = self._smart_compress_documents(docs, clean_query)

        return docs

    # ==================== [OPT-12] Token预算管理 ====================
    def _truncate_by_token_budget(self, docs: List[LangchainDocument]) -> List[LangchainDocument]:
        """
        按token预算截断检索结果，避免超出LLM上下文窗口。
        原理：LLM有最大context长度限制(默认4096 tokens≈3000中文)，
              超出部分会被截断，可能导致关键信息丢失。
              按相关性排序后优先保留高分chunk。
        """
        if not docs:
            return docs
        total_chars = sum(len(d.page_content) for d in docs)
        # 中文约1字符≈1.3token
        estimated_tokens = int(total_chars * 1.3)
        if estimated_tokens <= MAX_CONTEXT_TOKENS:
            return docs

        # 按顺序截断(已按相关性排序)
        budget_chars = int(MAX_CONTEXT_TOKENS / 1.3)
        kept = []
        used = 0
        for doc in docs:
            doc_len = len(doc.page_content)
            if used + doc_len <= budget_chars:
                kept.append(doc)
                used += doc_len
            else:
                remaining = budget_chars - used
                if remaining > 100:  # 至少保留100字符才有意义
                    truncated_doc = LangchainDocument(
                        page_content=doc.page_content[:remaining],
                        metadata={**doc.metadata, "truncated": True}
                    )
                    kept.append(truncated_doc)
                break
        return kept

    # ================== 文档压缩（LLM抽取式） ==================
    def _compress_document(self, document: LangchainDocument, query: str) -> LangchainDocument:
        """
        使用LLM对单个文档进行抽取式压缩
        提取与用户问题最相关的核心信息，去除冗余内容
        :param document: 原始文档对象
        :param query: 用户问题
        :return: 压缩后的文档对象
        """
        if not ENABLE_DOC_COMPRESSION or self.compression_model is None:
            return document

        original_content = document.page_content
        original_length = len(original_content)
        # 如果文档已经很短，不需要压缩
        target_len = COMPRESSION_TARGET_LENGTH
        if COMPRESSION_RATIO < 1.0:
            target_len = int(original_length * COMPRESSION_RATIO)

        if original_length <= target_len:
            return document

        # 构建压缩提示词模板
        compression_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是文档压缩助手。根据用户问题，从资料中提取最相关的核心信息。
压缩规则：
1. 严格保留原文，不添加原文没有的信息
2. 只提取与用户问题直接相关的内容
3. 保留重要的事实、数据、定义、结论
4. 去除重复、举例、过渡性语句
5. 控制输出长度约{target_len}字符
直接输出压缩后的文本。"""),
            ("human", """用户问题：{query}

原始资料：
{document_content}
""")
        ])
        try:
            # 执行压缩
            chain = compression_prompt | self.compression_model
            response = chain.invoke({
                "query": query,
                "document_content": original_content,
                "target_len": target_len
            })

            compressed = response.content.strip() if hasattr(response, 'content') else str(response).strip()

            # 如果压缩失败或压缩后为空，返回原文
            if not compressed or len(compressed) < 10:
                print(f"⚠️ 文档压缩失败，使用原文")
                return document

            compression_ratio = len(compressed) / original_length
            print(f"📦 文档压缩完成: {original_length} → {len(compressed)} 字符 (压缩率: {compression_ratio:.1%})")
            # 创建压缩后的新文档，保留原元数据并添加压缩信息
            compressed_doc = LangchainDocument(
                page_content=compressed,
                metadata={
                    **document.metadata,
                    "compressed": True,
                    "original_length": original_length,
                    "compressed_length": len(compressed)
                }
            )
            return compressed_doc
        except Exception as e:
            print(f"⚠️ 文档压缩出错: {e}，使用原文")
            return document

    # ================== 批量文档压缩 ==================
    def _compress_documents_batch(self, documents: list[LangchainDocument], query: str) -> list[LangchainDocument]:
        """
        批量压缩文档（可并行处理，这里先实现串行）

        :param documents: 原始文档列表
        :param query: 用户问题
        :return: 压缩后的文档列表
        """
        if not ENABLE_DOC_COMPRESSION or self.compression_model is None:
            return documents

        print(f"\n🗜️ 开始文档压缩 (共{len(documents)}个文档)...")
        tasks = []
        compressed_docs = []
        with ThreadPoolExecutor(max_workers = MAX_WORKERS) as executor:
            for i,doc in enumerate(documents):
                tasks.append(executor.submit(self._compress_document, doc, query))

            for task in as_completed(tasks):
                try:
                    compressed = task.result(timeout=30)
                    if compressed:
                        compressed_docs.append(compressed)
                except Exception as e:
                    print(f"⚠️ 文档异步压缩出错: {e}")
                    pass

        print(f"✅ 文档压缩完成\n")
        return compressed_docs

    # ================== 备用：句子级别过滤压缩 ==================
    def _compress_by_sentence_filtering(self, document: LangchainDocument, query: str, top_n_sentences: int = 5) -> LangchainDocument:
        """
        基于句子相关性过滤的压缩方法（不依赖LLM，速度快）
        使用简单的关键词匹配和位置权重选择最相关的句子
        :param document: 原始文档对象
        :param query: 用户问题
        :param top_n_sentences: 保留的最相关句子数量
        :return: 压缩后的文档对象
        """
        original_content = document.page_content
        # 分句（支持中英文标点）
        sentences = re.split(r'[。！？!?.\n]+', original_content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) <= top_n_sentences:
            return document

        # 提取查询关键词
        query_keywords = set(jieba.cut(query))
        query_keywords = {kw for kw in query_keywords if kw not in self._stopwords and len(kw) > 1}
        # 计算每个句子的相关性分数
        sentence_scores = []
        for i, sent in enumerate(sentences):
            # 关键词匹配得分
            sent_words = set(jieba.cut(sent))
            sent_words = {sw for sw in sent_words if sw not in self._stopwords and len(sw) > 1}
            keyword_match_count = len(query_keywords & sent_words)
            keyword_score = keyword_match_count / max(len(query_keywords), 1)

            # 位置得分（越靠前权重越高）
            position_score = 1.0 - (i / len(sentences)) * 0.5

            # 长度惩罚（过长的句子可能包含冗余）
            length_penalty = min(1.0, 100.0 / len(sent))

            total_score = keyword_score * 0.6 + position_score * 0.3 + length_penalty * 0.1
            sentence_scores.append((i, sent, total_score))
        # 选择得分最高的句子
        sentence_scores.sort(key=lambda x: x[2], reverse=True)
        selected_sentences = [s[1] for s in sentence_scores[:top_n_sentences]]

        # 按原始顺序重新排列
        selected_indices = set(s[0] for s in sentence_scores[:top_n_sentences])
        ordered_sentences = [sentences[i] for i in range(len(sentences)) if i in selected_indices]

        compressed_content = "。".join(ordered_sentences)

        print(f"✂️ 句子过滤压缩: {len(sentences)}句 → {len(ordered_sentences)}句")

        return LangchainDocument(
            page_content=compressed_content,
            metadata={
                **document.metadata,
                "compressed": True,
                "compression_method": "sentence_filtering"
            }
        )

    # ================== 智能压缩（自动选择方法） ==================
    def _smart_compress_documents(self, documents: list[LangchainDocument], query: str) -> list[LangchainDocument]:
        """
        智能文档压缩：根据文档长度和配置选择压缩策略
        - 使用LLM抽取式压缩（效果好但慢）
        - 或使用句子过滤压缩（快但效果略差）
        """
        if not ENABLE_DOC_COMPRESSION:
            return documents

        # 计算总长度
        total_length = sum(len(doc.page_content) for doc in documents)

        # 如果总长度已经不大，不需要压缩
        if total_length < COMPRESSION_TARGET_LENGTH * 2:
            print(f"📄 文档总长度({total_length}字符)较小，跳过压缩")
            return documents
        # 选择压缩方法
        # 这里默认使用LLM抽取式压缩，如果想用快速压缩可以切换
        use_llm_compression = True  # 可以改成配置项

        if use_llm_compression:
            return self._compress_documents_batch(documents, query)
        else:
            compressed = []
            for doc in documents:
                compressed.append(self._compress_by_sentence_filtering(doc, query))
            return compressed
