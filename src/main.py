"""
本地RAG知识库系统 - 主入口（流程调度）

使用方式：
1.将文档放入./RAG_files文件夹下
2.从项目根目录运行: python src/main.py （首次运行会自动建库）
3.向AI提问，它会基于你的文档回答问题

架构说明：
- LocalRAGSystem 通过多继承组合各功能模块（Mixin）：
    process_files.DocumentProcessMixin       文档加载/切分
    build_db.VectorDBMixin                   向量库构建/加载
    preprocess_query.QueryPreprocessMixin    查询预处理
    process_retrieval.RetrievalMixin         检索流程
    create_anwser.AnswerGenerateMixin        答案生成
    management_docs.DocumentManagementMixin  文档管理
- 本文件只保留：系统核心状态、模型懒加载属性、交互循环与命令行调度
"""

import argparse
import threading
import time
from typing import List, Set

#导入ollama模型的文本嵌入器、对话模式
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document as LangchainDocument
from langchain_core.callbacks import StreamingStdOutCallbackHandler
# ================== 多路召回 + 重排依赖 ==================
from sentence_transformers import CrossEncoder

import config
from config import (
    DOCS_DIR,
    MANIFEST_PATH,
    EMBED_MODEL,
    CHAT_MODEL,
    CHAT_NUM_CTX,
    OLLAMA_KEEP_ALIVE,
    RERANK_MODEL,
    ENABLE_MULTI_QUERY,
    MULTI_QUERY_MODEL,
    ENABLE_DOC_COMPRESSION,
    COMPRESSION_MODEL,
    COMPRESSION_TEMPERATURE,
    DEFAULT_TEST_SET,
)

from process_files import DocumentProcessMixin
from build_db import VectorDBMixin
from preprocess_query import QueryPreprocessMixin
from process_retrieval import RetrievalMixin
from create_anwser import AnswerGenerateMixin
from management_docs import DocumentManagementMixin, DocumentManifest
from evaluate import RAGEvaluator


class LocalRAGSystem(
    DocumentProcessMixin,      # 文档加载、文本处理与切分
    VectorDBMixin,             # 向量数据库构建与加载
    QueryPreprocessMixin,      # 查询预处理
    RetrievalMixin,            # 检索流程
    AnswerGenerateMixin,       # 答案生成
    DocumentManagementMixin,   # 文档管理
):
    def __init__(self):
        #懒加载配置
        self._embeddings = None
        self._chat_model = None
        self._multi_query_model = None
        self._compression_model = None
        self._rerank_model = None
        self.vector_store = None
        self.rag_chain = None
        self.chunks: List[LangchainDocument] = []
        self._bm25_index = None
        self._keyword_index = None
        self._stopwords: Set[str] = set()
        self._document_chain = None
        self._rag_prompt = None
        self._init_stopwords()
        self._manifest = DocumentManifest(MANIFEST_PATH)
        self._mutation_lock = threading.Lock()

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(
            model=EMBED_MODEL
            )
        return self._embeddings

    @property
    def chat_model(self):
        if self._chat_model is None:
            self._chat_model = self._create_chat_model()
        return self._chat_model

    def _create_chat_model(self, streaming: bool = True):
        """创建 ChatOllama 实例；streaming 仅交互模式需要"""
        kwargs = dict(
            model=CHAT_MODEL,
            temperature=0.3,
            num_threads=12,
            num_ctx=CHAT_NUM_CTX,
            #num_predict=CHAT_NUM_PREDICT,
            keep_alive=OLLAMA_KEEP_ALIVE,
            top_p=0.7,
            repeat_penalty=1.1,
            streaming=streaming,
        )
        if streaming:
            kwargs["callbacks"] = [StreamingStdOutCallbackHandler()]
        return ChatOllama(**kwargs)

    @property
    def rerank_model(self):
        if self._rerank_model is None:
            # [OPT-11] 自动检测设备，优先GPU
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            self._rerank_model = CrossEncoder(
                RERANK_MODEL, device=device
            )
        return self._rerank_model

    @property
    def multi_query_model(self):
        """[OPT-02修复] 改为property，原版是普通方法且逻辑错误"""
        if self._multi_query_model is None and ENABLE_MULTI_QUERY:
            self._multi_query_model = ChatOllama(
                model=MULTI_QUERY_MODEL,
                temperature=0.1,
                num_threads=12,
                num_ctx=4096,
                streaming=False,
            )
        return self._multi_query_model

    @property
    def compression_model(self):
        if self._compression_model is None and ENABLE_DOC_COMPRESSION:
            self._compression_model = ChatOllama(
                model=COMPRESSION_MODEL,
                temperature=COMPRESSION_TEMPERATURE,
                num_threads=12,
                num_ctx=2048,
                streaming=False,
            )
        return self._compression_model

    def _init_stopwords(self):
        self._stopwords = {
            '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
            '没有', '看', '好', '这个', '那个', '什么', '怎么', '为什么', '哪里',
            '哪个', '如何', '请问', '一下',
            '吧', '吗', '呢', '啊', '哦', '嗯', '哈哈', '是的', '好的', '可以', '应该',
            'a', 'an', 'the', 'and', 'of', 'to', 'in', 'for', 'on', 'with',
            'by', 'at', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had',
        }

    def run_interactive(self):
        """交互式问答"""
        print("\n"+"="*55)
        print("欢迎使用本地知识库问答系统")
        print("="*55)
        print("输入问题与知识库对话，输入quit退出")
        while True:
            try:
                question = input("\n请输入问题：\n").strip()
                if not question:
                    continue
                if question.lower() == 'quit':
                    print("谢谢使用，再见！")
                    break

                print("检索中...", end="", flush=True)
                start_time = time.time()
                response = self.ask_question(question)
                elapsed = time.time() - start_time
                print(f"\n耗时: {elapsed:.2f}秒")
            except KeyboardInterrupt:
                print("\n\n已中断，输入 quit 退出。")
            except Exception as e:
                print(f"\n发生错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="RAG知识库系统")
    parser.add_argument('--eval', action='store_true', help='运行评估模式')
    parser.add_argument('--testset', type=str, default=DEFAULT_TEST_SET, help='测试集路径')
    parser.add_argument('--limit', type=int, default=None, help='限制评估问题数量')
    parser.add_argument('--skip-build', action='store_true', help='跳过向量库构建')
    parser.add_argument('--no-warmup', action='store_true', help='跳过模型预热（首次问答会很慢）')
    parser.add_argument('--fast', action='store_true', help='快速模式：跳过重排，降低延迟')
    parser.add_argument('--perf', action='store_true', help='打印各阶段耗时明细')
    parser.add_argument('--ask', type=str, default=None, help='单次问答测试（例: --ask "什么是DCT"）')
    # 文档管理命令
    parser.add_argument('--add', type=str, default=None, help='添加单个文档到知识库')
    parser.add_argument('--remove', type=str, default=None, help='从知识库移除文档')
    parser.add_argument('--update', type=str, default=None, help='更新知识库中的文档')
    parser.add_argument('--sync', action='store_true', help='同步文档目录与知识库')
    parser.add_argument('--status', action='store_true', help='查看知识库文档状态')
    parser.add_argument('--watch', action='store_true', help='启动文件监控模式（自动同步文档变更）')
    parser.add_argument('--watch-interval', type=float, default=5.0, help='文件监控轮询间隔（秒）')
    parser.add_argument('--batch', type=str, default=None, help='批量添加文档目录（例: --batch ./docs）')

    args = parser.parse_args()
    if args.fast:
        config.FAST_MODE = True
        config.ENABLE_RERANK = False
    if args.perf:
        config.ENABLE_PERF_LOG = True

    rag = LocalRAGSystem()
    if not args.skip_build:
        rag.build_or_load_db()

    # 文档管理命令（无需向量库即可执行status）
    if args.status:
        if rag.vector_store is None:
            print("向量库未初始化")
        else:
            status = rag.get_document_status()
            print("\n" + "=" * 55)
            print("  知识库文档状态")
            print("=" * 55)
            m = status['manifest']
            print(f"  已索引文件: {m['total_documents']}")
            print(f"  已索引chunk: {m['total_chunks']}")
            print(f"  向量库向量数: {m['vector_count']}")
            print("-" * 55)
            for doc in status['documents']:
                hash_icon = "✓" if doc['hash_match'] else ("✗" if doc['hash_match'] is False else "?")
                disk_icon = "✓" if doc['on_disk'] else "✗"
                print(f"  [{hash_icon}] disk={disk_icon} {doc['file']} ({doc['chunks']} chunks)")
            print("=" * 55)
        return

    if rag.vector_store is None and not args.add:
        print("向量库为空，请先放入文档并运行，或使用 --add 添加文档")
        return

    if args.add:
        rag.add_document(args.add)
        return

    if args.remove:
        rag.remove_document(args.remove)
        return

    if args.update:
        rag.update_document(args.update)
        return

    if args.sync:
        rag.sync_knowledge_base()
        return

    if args.watch:
        if not args.no_warmup:
            rag.warmup_models(preload_chat=True)
        print("=" * 55)
        print("  文件监控模式")
        print(f"  监控目录: {DOCS_DIR}")
        print("  按 Ctrl+C 停止监控")
        print("=" * 55)
        watcher = rag.start_file_watcher(poll_interval=args.watch_interval)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n正在停止文件监控...")
            watcher._stop_event.set()
            watcher.join(timeout=10)
        return
    if args.batch:
        rag.add_documents_batch(args.batch)
        return
    if not args.no_warmup:
        rag.warmup_models(preload_chat=True)

    if args.ask:
        import time as _time
        t0 = _time.time()
        print(rag.ask_question(args.ask))
        print(f"\n单次问答总耗时: {_time.time() - t0:.1f}s")
        return

    if args.eval:
        print("n"+"="*55)
        print("进入评估模式"+"="*55)
        evaluator = RAGEvaluator(rag)
        summary = evaluator.run_evaluation(args.testset, limit=args.limit)
        print("\n评估完成！")
    else:
        rag.run_interactive()


if __name__ == '__main__':
    main()
