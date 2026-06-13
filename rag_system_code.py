"""
这是一个完整的本地RAG知识库系统
使用方式：
1.将文档放入./RAG_files文件夹下
2.运行rag_system.py文件，首次运行会自动建库
3.向AI提问，它会基于你的文档回答问题
"""

import os
from pathlib import Path
# 导入文档加载器
from langchain_community.document_loaders import(
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)

# 导入文本切分器
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
#导入ollama模型的文本嵌入器、对话模式
from langchain_ollama import OllamaEmbeddings, ChatOllama
# 导入向量数据库
from langchain_chroma import Chroma
#导入对话提示词模板
from langchain_core.prompts import ChatPromptTemplate
#导入langchain参数透传
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_unstructured import UnstructuredLoader
import re
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from langchain_core.documents import Document as LangchainDocument
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# ================== 多路召回 + 重排依赖 ==================
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np
import jieba

from langchain_core.callbacks import StreamingStdOutCallbackHandler
from concurrent.futures import ThreadPoolExecutor, as_completed


#==========配置项============
DOCS_DIR = '../RAG_files'   #文档目录
CHROMA_DB_DIR = '../chroma_db'   #向量数据库目录
EMBED_MODEL = 'nomic-embed-text-v2-moe'
CHAT_MODEL = 'qwen2.5:7b'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 10
RERANK_TOP_N = 5    # 重排后给LLM的数量
CLEAN_DOCS_DIR = '../RAG_files_clean'

SIMILARITY_THRESHOLD = 0.3          # 检索相似度阈值：低于该值直接判定为无有效资料
ENABLE_KEYWORD_AUGMENT = True      # 是否启用关键词增强
USE_LLM_FOR_KEYWORDS = False       # 是否使用LLM提取关键词（False则使用规则提取）
KEYWORD_EXTRACTION_MODEL = CHAT_MODEL  # 提取关键词使用的模型
MAX_KEYWORDS = 8                    # 最多提取多少个关键词
# 多Query检索配置
ENABLE_MULTI_QUERY = True           # 是否启用多Query检索
MULTI_QUERY_NUM = 3                 # 生成的额外Query数量
MULTI_QUERY_MODEL = CHAT_MODEL      # 生成多Query使用的模型
# ================== 文档压缩配置 ==================
ENABLE_DOC_COMPRESSION = True       # 是否启用文档压缩（LLM抽取式压缩）
COMPRESSION_TARGET_LENGTH = 300     # 压缩目标长度（字符数），每个文档压缩后的目标长度
COMPRESSION_RATIO = 0.5             # 压缩比例（如果设置此值，会覆盖target_length）
COMPRESSION_MODEL = CHAT_MODEL      # 压缩使用的模型
COMPRESSION_TEMPERATURE = 0.1       # 压缩时的温度（低温度保证提取准确性）
MAX_WORKERS = 4 #并发处理数
#=====================

class LocalRAGSystem:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=EMBED_MODEL
            )
        self.chat_model = ChatOllama(
            model=CHAT_MODEL, 
            temperature=0.3,
            num_threads=12,
            num_ctx = 4096,
            top_p = 0.7,
            repeat_penalty = 1.1,
            streaming = True,
            callbacks=[StreamingStdOutCallbackHandler()], #流式输出回调
            )
        # ==================多Query生成专用模型（低温度保证生成质量）=================
        self.multi_query_model = OllamaEmbeddings(
            model=MULTI_QUERY_MODEL
        ) if not ENABLE_MULTI_QUERY else ChatOllama(
            model=MULTI_QUERY_MODEL,
            temperature=0.1,  # 低温度减少随机性
            num_threads=12,
            num_ctx=4096,
            streaming=False    # 非流式输出
        )
        # ================== 文档压缩专用模型 ==================
        self.compression_model = ChatOllama(
            model=COMPRESSION_MODEL,
            temperature=COMPRESSION_TEMPERATURE,
            num_threads=12,
            num_ctx=2048,
            streaming=False    # 压缩时不需要流式输出
        ) if ENABLE_DOC_COMPRESSION else None

        self.vector_store=None
        self.rag_chain = None
        self.chunks = []
        # =================重排模型初始化=================
        self.rerank_model = CrossEncoder("./models/bge-reranker-base", device="cpu")
        # 这里先初始化一个轻量级的关键词提取器（基于jieba+TF-IDF）
        self._init_lightweight_keyword_extractor()
    
    def _init_lightweight_keyword_extractor(self):
        """初始化轻量级关键词提取器（不依赖LLM，纯规则+统计）"""
        # 加载停用词表（可以自己维护）
        self.stopwords = set([
            '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '这个', '那个', '什么', '怎么', '为什么', '哪里', '哪个', '如何', '请问', '一下',
            '吧', '吗', '呢', '啊', '哦', '嗯', '哈哈', '是的', '好的', '可以', '应该',
            'a', 'an', 'the', 'and', 'of', 'to', 'in', 'for', 'on', 'with', 'by', 'at',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'
        ])
       
    def _load_docx_with_heading(self,docx_path):
        """
        加载word文档转成markdown格式的文本。
        对标题 1, 2, 3 分别映射为 '#', '##', '###'。
        """
        word_doc = Document(docx_path)
        markdown_lines = []
        for para in word_doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
                
            # 获取段落样式对象
            para_style = para.style
            
            # 关键判断：样式的类型必须是段落样式，并且它的基础样式是“Normal”或直接就是内置标题
            # 内置标题样式名在WD_STYLE里没有直接常量，但可以通过type和name的规律或ID判断
            # 方法一：通过样式名称判断（适配中英文）
            style_name = para_style.name.lower() if para_style.name else ""
            
            # 方法二（最稳健）：通过样式ID判断。内置标题样式的ID类似于 'Heading1', 'Heading2'...
            # 注意：这是假设文档使用的是内置样式。如果用户自建了“标题1”，ID则不同。
            # 对于绝大多数文档，我们可以同时检查名称和ID的前缀。
            
            level = 0
            # 尝试从样式名称提取级别 (适配 'heading 1', '标题 1', 'head 1')
            # 检查标准格式 'Heading X' 或 '标题 X'
            match = re.search(r'heading (\d+)', style_name, re.IGNORECASE)
            if not match:
                match = re.search(r'标题 (\d+)', style_name)
            
            if match:
                level = int(match.group(1))
            else:
                # 实在找不到，按普通文本处理
                markdown_lines.append(text)
                continue
            
            # 只处理1-3级标题
            if level in [1, 2, 3]:
                prefix = '#' * level
                markdown_lines.append(f"{prefix} {text}")
            else:
                # 超出3级的标题，按普通文本处理，或可继续映射
                markdown_lines.append(text)
        
        full_markdown_text = "\n\n".join(markdown_lines)
        
        return LangchainDocument(
            page_content=full_markdown_text,
            metadata={"source": docx_path}
        )


    def _load_documents(self)->list:
        """
        加载目录下的所有文档
        :return: 文档列表
        """
        docs=[]
        docs_path=Path(DOCS_DIR)
        suffix_list = [".pdf", ".txt", ".docx", ".doc"]

        if not docs_path.exists():
            docs_path.mkdir(parents=True)
            print(f"已创建文档目录，{DOCS_DIR},请放入文档重新运行")
            return []

        # 加载txt文件
        for txt_file in docs_path.glob('*.txt'):
            try:
                loader = TextLoader(str(txt_file), encoding='utf-8')
                doc = loader.load()
                docs.extend(doc)
                print(f"已加载txt文件，{txt_file.name}")
            except Exception as e:
                print(f"加载txt文件{txt_file.name}失败，{str(e)}")

        # 加载pdf文件
        for pdf_file in docs_path.glob('*.pdf'):
            try:
                loader = PyPDFLoader(str(pdf_file))
                doc = loader.load()
                docs.extend(doc)
                print(f"已加载pdf文件，{pdf_file.name},共{len(doc)}页,docs大小：{len(docs)}")
            except Exception as e:
                print(f"加载pdf文件{pdf_file.name}失败，{str(e)}")
        #加载word文档
        for docx_file in docs_path.glob('*.docx'):
            docx_path = str(docx_file)
            doc = self._load_docx_with_heading(docx_path)
            docs.append(doc)
            print(f"已加载word文档，{docx_file.name}")

        # for suffix in suffix_list:
        #     file_path = list(docs_path.glob(f'*{suffix}'))
        #     for file in file_path:
        #         print(f"正在加载{suffix}文件，{file.name}")
        #         try:
        #             #使用UnstructuredLoader加载文档，必须安装对应的依赖包
        #             loader = UnstructuredLoader(
        #                 str(file),
        #                 mode = "elements", #开启精细元素解析
        #                 strategy = "hi_res", #高精度解析
        #                 extract_images_in_pdf=True,#开启，PDF 里图片的提取 + OCR 文字识别
        #                 pdf_infer_table_structure=True, #开启，自动识别PDF里面的表格，把表格解析成可读文本
        #                 extract_image_block_types=["Image"], #自动提取图片里的文字OCR识别
        #                 chunking_strategy="by_title",  # 自动按章节分块
        #                 include_page_breaks=False,        # 不保留分页符
        #                 languages=["chi_sim", "eng"],     # 支持中英文
        #             )
        #             doc = loader.load()
        #             docs.extend(doc)
        #             print(f"已加载{suffix}文件，{file.name}")

        #         except Exception as e:
        #             print(f"加载{suffix}文件{file.name}失败，{str(e)}")

        return docs

    def _split_documents(self,docs:list)->list:
        """
        对文档进行文本切分
        """
        final_chunks = []
        #word文档按照章节执行第一层切分
        for doc in docs:
            file_name = os.path.basename(doc.metadata['source'])
            
            #判断是否是word文档
            if file_name.endswith('.docx'):
                # 1. 设置第一层：基于标题的结构切分
                headers_to_split_on = [
                    ("#", "H1"),
                    ("##", "H2"),
                    ("###", "H3"),
                ]

                markdown_splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=headers_to_split_on
                )
                # 执行第一层切分
                header_splits = markdown_splitter.split_text(doc.page_content)
                # 2. 设置第二层：在章节内部进行语义切分（递归方法）
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    separators=['\n\n', '\n', '。 ', '. ', '！','？','!','?',' ',''],
                )
                # 执行第二层切分，得到最终的分块列表
                chunks = splitter.split_documents(header_splits)
                final_chunks.extend(chunks)
                print(f"文档{file_name}共分块{len(chunks)}个小块")
            #4. 补充源文件元数据到每个块
                for chunk in chunks:
                    if "source" not in chunk.metadata:
                        chunk.metadata["source"] = doc.metadata['source']
        # for i, chunk in enumerate(final_chunks[:20]):
        #     print(f"--- Chunk {i+1} ---")
        #     print(f"元数据: {chunk.metadata}")
        #     print(f"内容: {chunk.page_content[:200]}...\n")
        print(f"共分块{len(final_chunks)}个小块")
        return final_chunks

    def build_or_load_db(self)->None:
        """
        构建或加载向量数据库
        """
        if os.path.exists(CHROMA_DB_DIR):
            print(f"加载已有向量库，{CHROMA_DB_DIR}")
            """
            初始化一个Chroma向量库对象,存到变量vector_store中。创建或连接一个本地持久化的向量库，
            把文本转成向量后存在本地，用于后续检索问答
            """
            self.vector_store = Chroma(
                persist_directory=CHROMA_DB_DIR, # 向量数据库目录
                embedding_function=self.embeddings, #用什么模型转成向量
                collection_name='knowledge_base', # 向量库名称
            )
            count = self.vector_store._collection.count()
            print(f"向量加载完成，共{count}个向量")
            #从向量库中提取所有文档
            all_docs = self.vector_store.get()
            for i, (doc_id, text, metadata) in enumerate(zip(
                all_docs['ids'], 
                all_docs['documents'], 
                all_docs['metadatas']
            )):
                doc = LangchainDocument(
                    page_content=text,
                    metadata=metadata or {}
                )
                self.chunks.append(doc)
 
          
        else:
            print("首次运行，开始构建向量库")
            print(f"加载目录{DOCS_DIR}下的所有文档...")
            #加载文档
            docs =self._load_documents()
          
            if len(docs) == 0:
                print("目录下没有文档，无法构建向量库")
                return
            
            # 对文档进行文本切分
            self.chunks = self._split_documents(docs)
            # 为每个文档块添加索引元数据
            for i, chunk in enumerate(self.chunks):
                chunk.metadata["chunk_index"] = i  
            # # 构建向量库
            # 这一步会：
            #   - 自动为所有文本块调用 embeddings.embed_documents() 生成向量
            #   - 将向量和原始文本一起存入Chroma数据库
            self.vector_store = Chroma.from_documents(
                documents=self.chunks,
                persist_directory=CHROMA_DB_DIR, # 向量数据库目录
                embedding=self.embeddings,
                collection_name='knowledge_base',
            )
            print(f"向量库构建完成，共{len(self.chunks)}个向量")

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
            ("system", """你是一个专业的文档压缩助手。请根据用户的问题，从以下资料中提取最相关、最核心的信息。
            
压缩规则：
1. **严格保留原文**：只能从原文中摘取或概括，不能添加原文没有的信息
2. **聚焦问题**：只提取与用户问题直接相关的内容
3. **保留关键信息**：包括重要的事实、数据、定义、结论等
4. **去除冗余**：删除重复、举例、过渡性语句
5. **保持连贯**：压缩后的文本应该通顺、逻辑完整
6. **输出长度**：控制在{target_len}个字符左右
输出格式：
直接输出压缩后的文本，不要输出任何额外的解释或标记。"""),
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
            
            compressed_content = response.content.strip()
            
            # 如果压缩失败或压缩后为空，返回原文
            if not compressed_content or len(compressed_content) < 10:
                print(f"⚠️ 文档压缩失败，使用原文")
                return document
            
            compression_ratio = len(compressed_content) / original_length
            print(f"📦 文档压缩完成: {original_length} → {len(compressed_content)} 字符 (压缩率: {compression_ratio:.1%})")
            # 创建压缩后的新文档，保留原元数据并添加压缩信息
            compressed_doc = LangchainDocument(
                page_content=compressed_content,
                metadata={
                    **document.metadata,
                    "compressed": True,
                    "original_length": original_length,
                    "compressed_length": len(compressed_content)
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
                    compressed = task.result()
                    if compressed:
                        compressed_docs.append(compressed)
                except Exception as e:
                    print(f"⚠️ 文档异步压缩出错: {e}")

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
        query_keywords = {kw for kw in query_keywords if kw not in self.stopwords and len(kw) > 1}
        # 计算每个句子的相关性分数
        sentence_scores = []
        for i, sent in enumerate(sentences):
            # 关键词匹配得分
            sent_words = set(jieba.cut(sent))
            sent_words = {sw for sw in sent_words if sw not in self.stopwords and len(sw) > 1}
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
            ("system", """你是一个专业的查询扩展助手，需要根据用户的原始问题，生成{num}个不同角度的查询语句，用于提升文档检索的召回率。
生成规则：
1. 保持核心语义不变，仅从不同表述、不同角度、不同关键词组合生成
2. 每个查询简洁明了，长度控制在20字以内
3. 原始问题是英文则用英文，否则用中文
4. 只输出查询语句，每个查询占一行，禁止输出其他内容
5. 必须生成{num}个查询，不能多也不能少"""),
            ("human", "原始问题：{query}")
        ])
        
        try:
            # 构建生成链
            chain = multi_query_prompt | self.multi_query_model
            response = chain.invoke({
                "query": original_query,
                "num": MULTI_QUERY_NUM
            })
            
            # 解析生成的Query
            generated_queries = [q.strip() for q in response.content.split('\n') if q.strip()]
            # 去重并保留原始Query
            all_queries = list({original_query} | set(generated_queries))[:MULTI_QUERY_NUM + 1]
            
            print(f"\n🔍 多Query扩展结果：\n")
            for i, q in enumerate(all_queries):
                print(f"  Query {i+1}: {q}")
            
            return all_queries
        except Exception as e:
            print(f"⚠️ 多Query生成失败，使用原始查询: {e}")
            return [original_query]

    def _extract_keywords_llm(self, query: str) -> list[str]:
        """
        使用LLM提取查询关键词
        :param query: 用户原始问题
        :return: 关键词列表
        """
        keyword_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个关键词提取专家。请从用户问题中提取最核心的关键词，用于检索相关文档。
规则：
1. 只提取名词、专有名词、核心动词，忽略停用词（的、了、是、在等）
2. 每个关键词应该是独立且有意义的检索词
3. 关键词数量控制在{max_keywords}个以内
4. 只输出关键词，用逗号分隔，不要输出其他内容
"""),
            ("human", "{query}")
        ])
        
        try:
            chain = keyword_prompt | self.chat_model
            response = chain.invoke({
                "query": query,
                "max_keywords": MAX_KEYWORDS
            })
            # 解析LLM返回的关键词
            keywords = [kw.strip() for kw in response.content.split('，') if kw.strip()]
            # 如果LLM返回的是英文逗号分隔
            if len(keywords) <= 1 and ',' in response.content:
                keywords = [kw.strip() for kw in response.content.split(',') if kw.strip()]
            print(f"🔑 LLM提取的关键词: {keywords}")
            return keywords[:MAX_KEYWORDS]
        except Exception as e:
            print(f"⚠️ LLM关键词提取失败，回退到规则提取: {e}")
            return self._extract_keywords_rule_based(query)

    def _extract_keywords_rule_based(self, query: str) -> list[str]:
        """
        基于规则的关键词提取（轻量级，不依赖LLM）
        使用jieba分词 + 词性过滤 + TF-IDF/词频统计       
        :param query: 用户原始问题
        :return: 关键词列表
        """
        import jieba.posseg as pseg
        
        # 1. 分词并标注词性
        words = pseg.cut(query)
        
        # 2. 词性过滤：保留名词(n)、动名词(vn)、动词(v)、专有名词(nr, ns, nt)
        allowed_pos = ['n', 'vn', 'v', 'nr', 'ns', 'nt', 'eng']  # eng是英文单词
        
        candidate_words = []
        word_freq = {}
        
        for word, flag in words:
            word_lower = word.lower().strip()
            # 过滤条件：长度>1，不是纯停用词，词性符合要求
            if (len(word_lower) >= 2 and 
                word_lower not in self.stopwords and
                flag in allowed_pos):
                candidate_words.append(word_lower)
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        # 3. 如果没有有效的候选词，放宽条件：保留所有长度>=2且不是纯停用词的词
        if not candidate_words:
            words = jieba.cut(query)
            for word in words:
                word_lower = word.lower().strip()
                if len(word_lower) >= 2 and word_lower not in self.stopwords:
                    candidate_words.append(word_lower)
                    word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        # 4. 按词频排序，去重后返回
        unique_words = list(dict.fromkeys(candidate_words))  # 保持顺序去重
        # 优先返回高频词
        unique_words.sort(key=lambda x: word_freq.get(x, 0), reverse=True)
        
        keywords = unique_words[:MAX_KEYWORDS]
        print(f"🔑 规则提取的关键词: {keywords}")
        return keywords

    def _extract_keywords(self, query: str) -> list[str]:
        """
        统一的关键词提取接口
        根据配置选择使用LLM或规则提取
        """
        if USE_LLM_FOR_KEYWORDS:
            return self._extract_keywords_llm(query)
        else:
            return self._extract_keywords_rule_based(query)

    def _augment_query(self, query: str) -> tuple[str, list[str]]:
        """
        返回原始查询和提取的关键词列表
        """
        if not ENABLE_KEYWORD_AUGMENT:
            return query, []
        
        keywords = self._extract_keywords(query)
        print(f"📝 问题: {query}")
        print(f"🔑 提取关键词: {keywords}")
        
        # 可选：构建增强查询（将关键词拼接回去）
        # 但我们的多路召回会分别使用原始查询和关键词，所以这里只需要返回关键词
        return keywords
#===============query预处理=====================
    def _preprocess_query(self, query: str) -> str:
        """
        标准化Query预处理流程：
        1. 去除首尾空白字符
        2. 统一大小写（中文无影响，兼容英文）
        3. 去除特殊字符（保留中文、英文、数字、常用标点）
        4. 过滤停用词
        5. 修复半角/全角符号
        6. 去除冗余空格/换行
        """
        if not isinstance(query, str) or query.strip() == "":
            return ""
        original_query = query
        # 步骤1：基础清洗
        query = query.strip()  # 去首尾空白
        query = query.lower()  # 统一小写（英文）
        query = query.replace("\n", "").replace("\r", "").replace("\t", "")  # 去换行/制表符
        
        # 步骤2：修复全角/半角
        query = self._fix_full_half_width(query)
        
        # 步骤3：过滤特殊字符（保留中文、字母、数字、中文标点）
        # 正则说明：保留[\u4e00-\u9fa5]中文 | [a-zA-Z0-9]英文数字 | [，。！？；：""''（）【】]中文标点 .,!?[]()英文标点
        query = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''（）【】.,!?[]()]', '', query)
        
        # 步骤4： 去除常见的问候语和填充词
        greetings = [
            '你好', '您好', '请问', '我想问', '能不能', '可以告诉我', '麻烦问一下',
            '帮忙看看', '请教一下', '咨询一下', '了解一下', '想知道', '请问一下',
            'hi', 'hello', 'hey', 'excuse me', 'please', 'sorry'
        ]
        for greeting in greetings:
            if query.startswith(greeting):
                query = query[len(greeting):]
                break
        # 步骤5： 去除句首的"的"、"了"等无意义词
        query = re.sub(r'^[的了嘛啊哈噢哦呃]+', '', query)
        # 步骤6： 去除重复字符（例如"哈哈哈" -> "哈"，"。。。。" -> "。"）
        query = re.sub(r'(.)\1{2,}', r'\1', query)
        # 步骤7： 规范化空格：多个空格合并为一个
        query = re.sub(r'\s+', ' ', query)
        # 步骤8： 去除句尾无意义的语气词
        query = re.sub(r'[呢啊哈哦呃嘛啦]+$', '', query)
        # 步骤9：分词 + 停用词过滤
        words = jieba.lcut(query)  # 精确分词
        filtered_words = [word for word in words if word not in self.stopwords and len(word) > 0]
        
        # 步骤10：重组为干净的查询文本
        clean_query = "".join(filtered_words)

        # 步骤10： 如果清洗后为空，返回原查询
        if not clean_query or len(clean_query.strip()) < 2:
            print(f"⚠️ Query清洗后为空，使用原始查询")
            return original_query.strip()
        
        # 日志输出（可选）
        print(f"原始Query：{original_query} → 预处理后：{clean_query}")
        
        return clean_query

    def _fix_full_half_width(self, text: str) -> str:
        """修复全角/半角字符"""
        result = []
        for char in text:
            code = ord(char)
            # 全角转半角
            if 65281 <= code <= 65374:
                code -= 65248
            # 半角空格转全角
            elif code == 32:
                code = 12288
            result.append(chr(code))
        return "".join(result)

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
        query_words = {w for w in query_words if w not in self.stopwords}
        
        scored = []
        for idx, doc in enumerate(self.chunks):
            doc_words = set(jieba.cut(doc.page_content.strip()))
            doc_words = {w for w in doc_words if w not in self.stopwords}
            # 计算Jaccard相似度或简单计数
            intersection = len(query_words & doc_words)
            if intersection > 0:
                scored.append((idx, intersection))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:TOP_K]

    def _keyword_recall(self, query: str, topk=TOP_K):
        tokenized_query = list(jieba.cut(query.strip())) #对查询语句做分词，拆成关键词列表
        query_words = set(tokenized_query) # 对关键词列表去重
        scored = []
        for idx, doc in enumerate(self.chunks):
            doc_words = set(jieba.cut(doc.page_content.strip()))
            # 统计查询词和文档关键词的交集数量，即匹配的关键词数量
            cnt = len(query_words & doc_words)
            scored.append((idx, cnt)) # 记录文档索引和匹配的关键词数量
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
        
        tokenized_docs = []
        for d in self.chunks:
            tokens = list(jieba.cut(d.page_content.strip()))
            # 可选：过滤停用词
            tokens = [t for t in tokens if t not in self.stopwords]
            tokenized_docs.append(tokens)
        
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = list(jieba.cut(enhanced_query.strip()))
        tokenized_query = [t for t in tokenized_query if t not in self.stopwords]
        if not tokenized_query:
            return []
        
        scores = bm25.get_scores(tokenized_query)
        ranked = np.argsort(scores)[::-1][:TOP_K]
        return [(int(i), float(scores[i])) for i in ranked if scores[i] > 0]
        
    def _bm25_recall(self, query: str, topk=TOP_K):
        #对文档块做分词
        tokenized_docs = []
        for d in self.chunks:
            tokens = list(jieba.cut(d.page_content.strip()))
            tokenized_docs.append(tokens)
        # 初始化BM25模型
        bm25 = BM25Okapi(tokenized_docs)
        # 对查询做分词
        tokenized_query = list(jieba.cut(query.strip()))
        scores = bm25.get_scores(tokenized_query) # 计算查询与每个文档块的BM25分数
        ranked = np.argsort(scores)[::-1][:topk]
        return [(int(i), float(scores[i])) for i in ranked]

    #================= 向量检索召回 ==================
    def _vector_recall(self, query: str, topk=TOP_K):
        # 创建向量检索器
        # 注意：这里的query会由langchain自动调用embeddings.embed_query()进行向量化
        # embed_query方法内部会处理文本的清洗和分词
        ret = self.vector_store.as_retriever(
            search_type="similarity_score_threshold", # 按相似度分数阈值检索
            search_kwargs={
                "k": topk, # 返回前TOP_K个最相似的文本块
                "score_threshold": SIMILARITY_THRESHOLD}, 
            )
        # 执行检索
        docs = ret.invoke(query)
       # print(f"\n检索到，共{len(docs)}个文档块\n")
        #print(f"检索到的文档块： {docs}")
        result = []
        for doc in docs:
            # print(f"文档块内容：{doc.page_content}\n")
            # print(f"文档块元数据：{doc.metadata}\n")
            # print("=" * 50)
            # 获取真实索引
            idx = doc.metadata.get("chunk_index")
            #print(f"真实索引：{idx}")
            if idx is not None and idx >= 0 and idx < len(self.chunks):
                # 获取相似度分数
                score = doc.metadata.get("score", 0.0)
                result.append((int(idx), float(score)))
        return result
    #================== RRF融合 ==================
    def _rrf_fuse(self, lists, k=60):
        scores = {} # 初始化一个空字典，用于存储每个文档块的RRF分数，key是文档块索引，value是RRF分数
        #print(f"融合前候选：{len(lists)} 条\n")
        for lst in lists:
            #print(f"列表{lst}\n")
            #遍历列表里的每一条记录，enumerate(lst)给每个元素带上索引
            # rank：从0开始的排名，第一名是0，第二名是1
            # idx：列表里文档块索引
            # _：忽略，因为这里只需要索引
            for rank, (idx, _) in enumerate(lst):  
                #print(f"文档下标：{idx} | 排名：{rank}\n")
                s = 1.0 / (rank + k) # RRF核心公式，k是平滑参数，防止除0错误
                # 把当前item的RRF分数累加到总字典里，如果idx存在字典里，分数就累加，否则默认是0。
                # 这里已经实现了去重，因为idx是文档的唯一ID，scores字典的key是唯一的，idx是字典里的key值
                scores[idx] = scores.get(idx, 0) + s 
        return sorted(scores.items(), key=lambda x: x[1], reverse=True) # 按RRF分数从高到低排序

    #================== 重排 ==================
    def _cross_encoder_rerank(self, query: str, fused_indices: list):
        # 从融合后的候选列表中提取文档块内容，返回的是列表格式
        candidates = [self.chunks[idx].page_content for idx, _ in fused_indices]
        # 从融合后的候选列表中提取文档块对象，返回的是列表格式
        docs_obj = [self.chunks[idx] for idx, _ in fused_indices]
        # cross_encoder重排序的固定格式，把用户问题和每一条候选文本组成一对
        pairs = [[query, c] for c in candidates]
        # 重排序模型打分，对每个问题-候选文本对进行打分，输出相关性分数
        scores = self.rerank_model.predict(pairs).tolist()
        # 把文档块对象和相关性分数打包在一起，按相关性分数从高到低排序
        reranked = sorted(zip(docs_obj, scores), key=lambda x: x[1], reverse=True)
        return reranked[:RERANK_TOP_N] #返回排名最靠前的RERANK_TOP_N个结果

# ====================== 单Query检索 ======================
    def _single_query_retrieve(self, query: str) -> list:
        """单Query的三路召回"""
        # 步骤2：查询增强 - 提取关键词
        augmented_keywords = self._augment_query(query)
        
        # 步骤3：三路召回
        bm25 = self._bm25_recall_with_augment(query , augmented_keywords)
        vec = self._vector_recall(query)
        kw = self._keyword_recall_with_augment(query, augmented_keywords)
        
        return [bm25, vec, kw]
# ====================== 多Query检索======================
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
        
        # 步骤2：每个Query独立检索
        all_retrieval_results = []
        for i, query in enumerate(all_queries):
            print(f"\n==== 处理扩展Query {i+1}/{len(all_queries)} ====")
            single_results = self._single_query_retrieve(query)
            all_retrieval_results.extend(single_results)
        
        # 步骤3：融合所有检索结果
        print(f"\n==== 融合所有Query的检索结果 ====")
        fused = self._rrf_fuse(all_retrieval_results)
        print(f"多Query融合后候选：{len(fused)} 条")

        # 步骤4：CrossEncoder重排
        print("==== CrossEncoder 重排 ====")
        reranked = self._cross_encoder_rerank(clean_query, fused)

        final = []
        for i, (doc, score) in enumerate(reranked):
            print(f"Top{i+1} | 相关性：{score:.4f} | {doc.page_content[:60]}...")
            final.append(doc)
        
        return final
    # ====================== 最终检索（三路→RRF→重排）======================
    def _retrieve_with_rerank(self, query: str) -> list[LangchainDocument]:
        """
        根据配置选择单Query检索或多Query检索。
        核心检索方法：
        1. 对查询做基础清洗
        2. 提取关键词增强查询
        3. 三路召回（向量 + BM25 + 关键词）
        4. RRF融合
        5. CrossEncoder重排
        """
        final = []
        # 步骤1：对query做基础清洗
        clean_query = self._preprocess_query(query)
        if ENABLE_MULTI_QUERY:
            final = self._multi_query_retrieve(clean_query)
        else:
            single_retrieve_result = self._single_query_retrieve(clean_query)
            fused = self._rrf_fuse(single_retrieve_result)
            print(f"融合后候选：{len(fused)} 条")

            print("==== CrossEncoder 重排 ====")
            reranked = self._cross_encoder_rerank(clean_query, fused)

            
            for i, (doc, score) in enumerate(reranked):
                print(f"Top{i+1} | 相关性：{score:.4f} | {doc.page_content[:60]}...")
                final.append(doc)
        final_compressed = self._smart_compress_documents(final, clean_query)
        return final_compressed

    def _build_rag_chain(self):
        """
        构建RAG链，使用重排结果
        """
        #创建提示词模板
        RAG_PROMPT = ChatPromptTemplate.from_messages([
            ("system","""请严格按照以下规则回答。
            规则（必须严格执行）：
            - 只看【参考资料】，不许用自己的知识
            - 若【参考资料】无对应内容，只说：我在知识库中未找到相关信息
            - 绝对不能猜测、编造、补充额外内容

            【参考资料】：{context}
            """),
            ("human","问题：{input}")
               ])
        
        def retrieve(inputs):
            q = inputs["input"]
            docs = self._retrieve_with_rerank(q)
            return {"input": q, "context": docs}

        
        #创建语义检索器
        # 当用户提问时，LangChain会自动：
        #   - 调用 embeddings.embed_query() 将问题转为向量
        #   - 在向量数据库中搜索最相似的文本块
        # retriever = self.vector_store.as_retriever(
        #     #search_type="similarity", # 按相似度检索
        #     #search_kwargs={"k": TOP_K} # 返回前TOP_K个最相似的文本块
        #     search_type="similarity_score_threshold", # 按相似度分数阈值检索
        #     search_kwargs={
        #         "k": TOP_K, # 返回前TOP_K个最相似的文本块
        #         "score_threshold": SIMILARITY_THRESHOLD}, 
        #     )
   
        
        #创建RAG链
        # 2. 创建文档组合链
        document_chain = create_stuff_documents_chain(self.chat_model, RAG_PROMPT)
        def run_chain(inputs):
            return document_chain.invoke(inputs)

        # 3. 创建检索链
        #self.rag_chain = create_retrieval_chain(retriever, document_chain)
        self.rag_chain = retrieve | RunnableLambda(run_chain)

    def ask_question(self,question:str)->str:
        """向知识库提问"""
        if self.vector_store is None:
            return "错误，向量库未初始化"
        if self.rag_chain is None:
            self._build_rag_chain()
        response = self.rag_chain.invoke({"input": question})#新版写法必须传字典，key固定是input
        # ====================== 打印检索到的文档 ======================
        # print("=" * 50)
        # print("🔍 检索到的上下文文档：")
        # for i, doc in enumerate(response["context"]):
        #     print(f"\n--- 文档 {i+1} ---")
        #     print(doc.page_content)
        # print("=" * 50)
        # ==============================================================
        return response #返回答案

    
    def add_document(self,file_path:str)->None:
        """添加文档到现有知识库"""
        if file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            loader = PyPDFLoader(file_path)

        docs = loader.load()
        # 对文档进行文本切分
        chunks = self._split_documents(docs)
        # 构建向量库
        self.vector_store.add_documents(chunks)
        print(f"已添加文档，{file_path}，共{len(chunks)}个文本块")

    def run_interactive(self):
        """交互式问答"""
        print("\n"+"="*55)
        print("欢迎使用本地知识库问答系统")
        print("="*55)
        print("输入问题与知识库对话，输入quit退出")
        while True:
            question = input("请输入问题：\n").strip()
            if not question:
                continue
            if question.lower() == 'quit':
                print("谢谢使用，再见！")
                break
            print("\n检索中，请稍后...",end="", flush=True)
            import time
            start_time = time.time()
            response = self.ask_question(question)
            elapsed_time = time.time() - start_time
            print(f"检索耗时：{elapsed_time:.2f}秒")
            print(f"\n知识库回复：\n{response}\n\n")

def main():
    rag = LocalRAGSystem()
    rag.build_or_load_db()
    if rag.vector_store:
        rag.run_interactive()
    
if __name__ == '__main__':
    main()
