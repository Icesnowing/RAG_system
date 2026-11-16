"""
答案生成模块
对应原 rag_system_code.py 中 LocalRAGSystem 的：
    _get_rag_prompt / _ensure_document_chain / _generate_answer / _build_rag_chain / ask_question
将检索结果与用户查询一起喂给大模型生成回答。
以 Mixin 形式提供给主系统类使用，方法实现与原文件保持一致。
"""

import time
from typing import List

#导入对话提示词模板
from langchain_core.prompts import ChatPromptTemplate
#导入langchain参数透传
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document as LangchainDocument
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

import config


class AnswerGenerateMixin:
    """RAG 链构建与答案生成能力（供 LocalRAGSystem 多继承组合）"""

    def _get_rag_prompt(self):
        """复用Prompt模板，避免重复构建"""
        if self._rag_prompt is None:
            self._rag_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个专业的技术文档问答助手。请严格按以下规则回答：

【核心规则 - 必须严格遵守】
1. 只能基于【参考资料】中的内容回答，绝对禁止使用你自己的知识
2. 如果【参考资料】中没有相关信息，直接回答："根据所提供的资料，无法回答此问题。"
3. 禁止猜测、编造、推理、补充任何参考资料中没有的信息
4. 如果参考资料中有答案，请完整、准确地复述相关内容

【回答格式要求】
- 对于事实查询：简洁直接地给出答案，关键数据必须准确
- 对于总结归纳：分点列出要点，保持逻辑顺序
- 对于对比分析：先分别描述各方特点，再总结差异
- 如需引用资料中的具体数值或公式，请原样保留

【参考资料】
{context}"""),
                ("human", "{input}")
            ])
        return self._rag_prompt

    def _ensure_document_chain(self):
        """懒加载文档组合链，供生成阶段复用"""
        if self._document_chain is None:
            self._document_chain = create_stuff_documents_chain(
                self.chat_model, self._get_rag_prompt()
            )

    def _generate_answer(self, question: str, docs: List[LangchainDocument]) -> str:
        """基于已检索文档直接生成答案，跳过重复检索"""
        self._ensure_document_chain()
        response = self._document_chain.invoke({"input": question, "context": docs})
        return response if isinstance(response, str) else str(response)

    def _build_rag_chain(self):
        """构建 RAG 链（仅组装 Runnable，不加载 LLM，耗时 <1ms）"""
        if self.rag_chain is not None:
            return

        def retrieve(inputs):
            q = inputs["input"]
            docs = self._retrieve_with_rerank(q)
            return {"input": q, "context": docs}

        def run_chain(inputs):
            self._ensure_document_chain()
            return self._document_chain.invoke(inputs)

        self.rag_chain = RunnableLambda(retrieve) | RunnableLambda(run_chain)

    def ask_question(self, question: str) -> str:
        """向知识库提问"""
        if self.vector_store is None:
            return "错误：向量库未初始化，请先放入文档并构建向量库。"

        if not question or not question.strip():
            return "请输入有效的问题。"

        if self.rag_chain is None:
            try:
                self._build_rag_chain()
            except Exception as e:
                return f"错误：RAG链初始化失败 - {str(e)}"

        try:
            t0 = time.time()
            t_ret, t_gen = t0, t0
            if config.ENABLE_PERF_LOG:
                clean = self._preprocess_query(question)
                t_ret = time.time()
                docs = self._retrieve_with_rerank(clean or question)
                print(f"  检索: {time.time() - t_ret:.2f}s")
                t_gen = time.time()
                response = self._generate_answer(question, docs)
                print(f"  生成: {time.time() - t_gen:.2f}s")
                print(f"  总计: {time.time() - t0:.2f}s")
            else:
                response = self.rag_chain.invoke({"input": question})
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            return f"错误：处理问题时发生异常 - {str(e)}"
