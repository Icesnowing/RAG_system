"""
全局配置与共享工具模块
所有配置项取值与原 rag_system_code.py 完全一致，供各功能模块统一引用。
注意：DOCS_DIR / CHROMA_DB_DIR / RERANK_MODEL 等相对路径基于「运行时工作目录」，
     请从项目根目录运行（python src/main.py），行为与原脚本一致。
"""

import os
import time
from functools import wraps

import jieba


# ==================异常重试装饰器 ==================
def retry_on_failure(max_retries: int = 2, delay: float = 1.0, fallback_value=None):
    """
    对模型调用进行自动重试，防止偶发网络/模型错误导致流程中断。
    原理：LLM推理是概率性的，偶发超时或OOM不应中断批量评估；
          重试+退避可以恢复绝大多数瞬态错误。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(delay * (attempt + 1))
                    else:
                        if fallback_value is not None:
                            return fallback_value
                        raise last_exception
            return fallback_value
        return wrapper
    return decorator


#==========配置项============
DOCS_DIR = './RAG_files'   #文档目录（项目根目录下的 RAG_files）
CHROMA_DB_DIR = './chroma_db'   #向量数据库目录
MANIFEST_PATH = os.path.join(CHROMA_DB_DIR, 'document_manifest.json')  #文档清单文件
EMBED_MODEL = 'nomic-embed-text-v2-moe'
CHAT_MODEL = 'qwen2.5:7b'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 10
RERANK_TOP_N = 5     # 重排后给LLM的数量（3条通常足够，5条会显著增加生成耗时）
CLEAN_DOCS_DIR = './RAG_files_clean'
DEFAULT_TEST_SET = 'rag_testsets.csv'
EVAL_RESULTS_DIR = 'eval_results'
RERANK_MODEL = "./models/bge-reranker-base"

SIMILARITY_THRESHOLD = 0.3          # 检索相似度阈值：低于该值直接判定为无有效资料
ENABLE_KEYWORD_AUGMENT = False      # 是否启用关键词增强
USE_LLM_FOR_KEYWORDS = False       # 是否使用LLM提取关键词（False则使用规则提取）
KEYWORD_EXTRACTION_MODEL = CHAT_MODEL  # 提取关键词使用的模型
MAX_KEYWORDS = 8                    # 最多提取多少个关键词
KEYWORD_MAX_DF_RATIO = 0.3          # 倒排索引高频词剪枝阈值：出现在超过此比例文档中的词跳过
# 多Query检索配置
ENABLE_MULTI_QUERY = False           # 是否启用多Query检索
MULTI_QUERY_NUM = 3                 # 生成的额外Query数量
MULTI_QUERY_MODEL = CHAT_MODEL      # 生成多Query使用的模型
# ================== 文档压缩配置 ==================
ENABLE_DOC_COMPRESSION = False       # 是否启用文档压缩（LLM抽取式压缩）
COMPRESSION_TARGET_LENGTH = 300     # 压缩目标长度（字符数），每个文档压缩后的目标长度
COMPRESSION_RATIO = 0.5             # 压缩比例（如果设置此值，会覆盖target_length）
COMPRESSION_MODEL = CHAT_MODEL      # 压缩使用的模型
COMPRESSION_TEMPERATURE = 0.1       # 压缩时的温度（低温度保证提取准确性）
MAX_WORKERS = 4 #并发处理数
#=====================

# [OPT-14] 耗时优化配置
RERANK_CANDIDATE_LIMIT = 15      # RRF融合后送入重排模型的最大候选数（减少CrossEncoder计算量）
RERANK_BATCH_SIZE = 16           # CrossEncoder批推理大小
ENABLE_PARALLEL_RECALL = True    # 三路召回是否并行执行
DEDUP_METHOD = "text"            # 去重方式: "text"(快速字符相似度) 或 "embedding"(慢但更准)
TEXT_DEDUP_THRESHOLD = 0.75      # 文本去重相似度阈值

# [OPT-15] 模型冷启动与推理加速（190s 主要来自 Ollama 首次加载 + LLM 生成）
ENABLE_RERANK = True             # False 时跳过重排，仅用 RRF Top-N（约省 5-15s/次）
PRELOAD_MODELS = True            # 启动时预热 embedding/rerank/chat，避免首次问答卡顿
OLLAMA_KEEP_ALIVE = "60m"        # 保持 Ollama 模型常驻内存，避免每次重新加载
CHAT_NUM_CTX = 4096              # 降低上下文窗口分配（原4096），加快推理
CHAT_NUM_PREDICT = 300           # 限制最大生成长度，避免冗长输出拖慢响应
CHAT_STREAMING = True           # 流式输出会略增开销；交互模式仍可见完整回答
ENABLE_PERF_LOG = False          # True 时打印各阶段耗时明细
FAST_MODE = False                # True = 跳过重排，检索更快但精度略降

# [OPT-08扩展] 上下文窗口管理
MAX_CONTEXT_TOKENS = 3000        # 送给LLM的最大context token数（越小生成越快）
DEDUP_THRESHOLD = 0.85           # chunk去重相似度阈值 [OPT-07]（仅embedding模式使用）
ENABLE_DEDUP = True              # 是否启用chunk去重 [OPT-07]

# 预热jieba分词，避免首次查询时的冷启动延迟
jieba.initialize()
