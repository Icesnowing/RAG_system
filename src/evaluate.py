"""
评估体系模块
对应原 rag_system_code.py 中的：
    EvalResult 数据类 / EnterpriseJudge 企业级评估器 / RAGEvaluator 评估流程
代码实现与原文件保持一致。
"""

import os
import re
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List

import jieba
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import EVAL_RESULTS_DIR, retry_on_failure


@dataclass
class EvalResult:
    """评估结果数据类"""
    question: str
    ground_truth: str
    retrieved_answer: str
    retrieved_contexts: list[str]
    question_type: str
    difficulty: str

    # 评估指标
    answer_relevance: float = 0.0
    answer_accuracy: float = 0.0
    faithfulness: float = 0.0
    context_recall: float = 0.0

    # [OPT-04新增] 子维度指标，便于细粒度分析
    fact_coverage: float = 0.0        # 事实覆盖度（ground_truth中的关键事实被答案覆盖的比例）
    hallucination_rate: float = 0.0   # 幻觉率（答案中无法从context验证的比例）

    # 性能指标
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # 检索质量
    retrieved_count: int = 0
    has_relevant_context: bool = False
    def to_dict(self):
        return {
            'question': self.question,
            'ground_truth': self.ground_truth,
            'retrieved_answer': self.retrieved_answer[:500] + '...' if len(self.retrieved_answer) > 500 else self.retrieved_answer,
            'question_type': self.question_type,
            'difficulty': self.difficulty,
            'answer_relevance': round(self.answer_relevance, 4),
            'answer_accuracy': round(self.answer_accuracy, 4),
            'faithfulness': round(self.faithfulness, 4),
            'context_recall': round(self.context_recall, 4),
            'fact_coverage': round(self.fact_coverage, 4),
            'hallucination_rate': round(self.hallucination_rate, 4),
            'retrieval_time_ms': round(self.retrieval_time_ms, 2),
            'generation_time_ms': round(self.generation_time_ms, 2),
            'total_time_ms': round(self.total_time_ms, 2),
            'retrieved_count': self.retrieved_count,
            'has_relevant_context': self.has_relevant_context,
        }


# ================== [OPT-04] 企业级评估器 ==================
class EnterpriseJudge:
    """
    企业级评估器 - 从关键词重叠升级为语义级评估

    原理说明：
    - 关键词重叠(Jaccard)忽略语义，2个句子可能0重叠但语义相同("DCT变换" vs "离散余弦变换")
    - 企业级评估需要: 语义相似度(embedding) + 事实匹配(cross-encoder) + 结构匹配(ROUGE)
    - 耗时优化: 优先使用轻量算法，仅在必要时使用LLM/cross-encoder
    """

    def __init__(self, embeddings=None, rerank_model=None):
        """
        :param embeddings: OllamaEmbeddings 实例，用于语义相似度计算
        :param rerank_model: CrossEncoder 实例，用于精细相关性判断
        """
        self._embeddings = embeddings
        self._rerank_model = rerank_model
        self._stopwords = self._init_stopwords()

    def _init_stopwords(self) -> set[str]:
        return {
            '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
            '没有', '看', '好', '这个', '那个', '什么', '怎么', '为什么', '哪里',
            '哪个', '如何', '请问', '一下', '吧', '吗', '呢', '啊', '哦', '嗯',
            '是的', '好的', '可以', '应该',
            'a', 'an', 'the', 'and', 'of', 'to', 'in', 'for', 'on', 'with',
            'by', 'at', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had',
        }

    # ------------------ 辅助方法 ------------------
    def _tokenize_keywords(self, text: str) -> set[str]:
        """提取文本中的非停用词集合"""
        words = set(jieba.cut(text))
        return {w for w in words if len(w) > 1 and w not in self._stopwords}

    def _split_sentences(self, text: str) -> list[str]:
        """智能分句，支持中英文标点"""
        if not text:
            return []
        raw = re.split(r'[。！？!?\n；;]+', text)
        return [s.strip() for s in raw if len(s.strip()) >= 4]

    def _extract_key_facts(self, text: str) -> list[str]:
        """
        从文本中提取关键事实点（以句子为单位）
        原理：评估召回率时，不是看词覆盖，而是看关键事实是否被检索到
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= 3:
            return sentences
        # 按长度和位置加权选择关键句
        scored = []
        for i, sent in enumerate(sentences):
            keywords = self._tokenize_keywords(sent)
            score = len(keywords) * (1.0 - 0.3 * (i / max(len(sentences), 1)))  # 位置权重
            scored.append((sent, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        # 取前一半或至少3句
        n = max(3, len(sentences) // 2)
        return [s[0] for s in scored[:n]]

    def _cosine_similarity(self, vec1, vec2) -> float:
        """计算向量余弦相似度"""
        if vec1 is None or vec2 is None:
            return 0.0
        dot = np.dot(vec1, vec2)
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return float(dot / norm) if norm > 0 else 0.0

    def _embed_text(self, text: str):
        """获取文本的embedding向量，带异常保护；同文本复用缓存避免重复调用"""
        if self._embeddings is None or not text or not text.strip():
            return None
        cache_key = text[:2000]
        if not hasattr(self, '_embed_cache'):
            self._embed_cache = {}
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]
        try:
            vec = np.array(self._embeddings.embed_query(text))
            if len(self._embed_cache) < 64:
                self._embed_cache[cache_key] = vec
            return vec
        except Exception:
            return None

    def _compute_rouge_l(self, reference: str, candidate: str) -> float:
        """
        计算 ROUGE-L (Longest Common Subsequence) F1
        原理：LCS衡量答案与标准答案的结构相似度，比Jaccard更能反映内容正确性
        """
        if not reference or not candidate:
            return 0.0
        # 使用字符级别的LCS，对中文更友好
        ref_chars = list(reference)
        cand_chars = list(candidate)
        m, n = len(ref_chars), len(cand_chars)
        if m == 0 or n == 0:
            return 0.0
        # 动态规划求LCS长度
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_chars[i - 1] == cand_chars[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs_len = dp[m][n]
        recall = lcs_len / m if m > 0 else 0
        precision = lcs_len / n if n > 0 else 0
        if recall + precision == 0:
            return 0.0
        return 2 * recall * precision / (recall + precision)

    # ==================== 四大核心指标 ====================

    def compute_faithfulness(self, question: str, answer: str, contexts: list[str]) -> float:
        """
        [OPT-04] 企业级忠实度计算

        原理：
        - 原版：答案逐句与context做词重叠，30%即认为"支持"，阈值过低，导致虚假高分。
        - 优化：三阶段混合策略：
          Step1 (快速): 句子级关键词Jaccard预筛(阈值50%)，快速排除明显不支持的句子
          Step2 (中等): 对边界case用embedding语义相似度验证(阈值0.55)
          Step3 (精细): 当有cross-encoder时做深度校验

        最终忠实度 = 被支持的句子数 / 总句子数
        额外输出 hallucination_rate = 1 - faithfulness（幻觉率）
        """
        answer_sentences = self._split_sentences(answer)
        if not answer_sentences:
            return 1.0

        all_context = " ".join(contexts)
        context_words = self._tokenize_keywords(all_context)
        ctx_vec = self._embed_text(all_context[:1000])  # 上下文向量只算一次

        supported = 0
        for sent in answer_sentences:
            sent_words = self._tokenize_keywords(sent)
            if not sent_words:
                continue

            # Step1: 关键词Jaccard（阈值从0.3提升到0.5）
            intersection = len(sent_words & context_words)
            jaccard = intersection / len(sent_words) if sent_words else 0

            if jaccard >= 0.5:  # [OPT-04] 从0.3提升到0.5
                supported += 1
            elif jaccard >= 0.2:
                # Step2: 边界case用语义相似度二次校验
                sent_vec = self._embed_text(sent)
                semantic_score = self._cosine_similarity(sent_vec, ctx_vec)
                if semantic_score >= 0.55:
                    supported += 1
                elif self._rerank_model is not None and jaccard >= 0.1:
                    # Step3: CrossEncoder精细校验（最慢但最准）
                    try:
                        pair_score = self._rerank_model.predict([[sent, all_context[:1500]]])[0]
                        if pair_score > 0.3:
                            supported += 1
                    except Exception:
                        pass
            # jaccard < 0.2: 明确不支持，不计数

        return supported / len(answer_sentences)

    def compute_context_recall(self, ground_truth: str, contexts: list[str]) -> float:
        """
        [OPT-04] 企业级上下文召回率

        原理：
        - 原版：ground_truth词集合 vs context词集合的Jaccard
          问题：Jaccard不区分关键词重要程度，"的"和"DCT"权重一样。
        - 优化：关键事实覆盖法
          Step1: 从ground_truth中提取关键事实点(按句子+位置权重)
          Step2: 每个关键事实点与context计算语义相似度
          Step3: 相似度>阈值的视为"已检索到"

        影响：能精确衡量"用户想知道的那些关键信息，检索系统是否找到了"
        """
        all_context = " ".join(contexts)
        if not ground_truth or not all_context:
            return 0.0
        if len(ground_truth.strip()) < 10:
            return 1.0

        # Step1: 提取关键事实
        key_facts = self._extract_key_facts(ground_truth)
        if not key_facts:
            gt_words = self._tokenize_keywords(ground_truth)
            ctx_words = self._tokenize_keywords(all_context)
            covered = len(gt_words & ctx_words)
            total = len(gt_words)
            return covered / total if total > 0 else 1.0

        # Step2: 每个关键事实与context做语义匹配
        ctx_vec = self._embed_text(all_context[:1500])  # 上下文向量只算一次
        ctx_words = self._tokenize_keywords(all_context)
        covered_facts = 0
        for fact in key_facts:
            fact_vec = self._embed_text(fact)
            sim = self._cosine_similarity(fact_vec, ctx_vec)

            if sim >= 0.45:  # 语义阈值：中等相关即可认为检索到
                covered_facts += 1
            else:
                # 快速词重叠作为兜底
                f_words = self._tokenize_keywords(fact)
                word_overlap = len(f_words & ctx_words) / max(len(f_words), 1)
                if word_overlap >= 0.4:
                    covered_facts += 1

        return covered_facts / len(key_facts)

    def compute_answer_accuracy(self, answer: str, ground_truth: str) -> float:
        """
        [OPT-04] 企业级答案准确性

        原理：
        - 原版：answer vs ground_truth的Jaccard
          问题："DCT(离散余弦变换)"和"DCT"词重叠低但语义一致
        - 优化：三维度加权融合
          (1) 语义相似度 (embedding cos) - 权重0.4
          (2) 关键事实匹配 (事实覆盖) - 权重0.35
          (3) ROUGE-L (结构匹配) - 权重0.25

        三维度互补：
        - 语义相似度处理"说法不同但意思相同"
        - 事实匹配确保核心信息点不遗漏
        - ROUGE-L捕捉答案结构的完整度
        """
        if not answer or not ground_truth:
            return 0.0
        if len(ground_truth.strip()) < 5:
            return 1.0

        # (1) 语义相似度
        ans_vec = self._embed_text(answer)
        gt_vec = self._embed_text(ground_truth)
        semantic_sim = self._cosine_similarity(ans_vec, gt_vec)

        # (2) 关键事实匹配：ground_truth中的关键事实是否在答案中出现
        key_facts = self._extract_key_facts(ground_truth)
        a_words = self._tokenize_keywords(answer)
        matched = 0
        for fact in key_facts:
            if ans_vec is not None:
                f_vec = self._embed_text(fact)
                if f_vec is not None:
                    sim = self._cosine_similarity(f_vec, ans_vec)
                    if sim >= 0.4:
                        matched += 1
                        continue
            # 快速词重叠兜底
            f_words = self._tokenize_keywords(fact)
            overlap = len(f_words & a_words) / max(len(f_words), 1)
            if overlap >= 0.35:
                matched += 1
        fact_match = matched / len(key_facts) if key_facts else semantic_sim

        # (3) ROUGE-L
        rouge = self._compute_rouge_l(ground_truth, answer)

        # 加权融合
        accuracy = 0.40 * semantic_sim + 0.35 * fact_match + 0.25 * rouge
        return min(1.0, accuracy)

    def compute_answer_relevance(self, answer: str, question: str) -> float:
        """
        [OPT-04] 企业级答案相关性

        原理：
        - 原版：answer vs question词重叠
          问题：答案包含很多技术术语但完全未回答用户问题也能得高分
        - 优化：双维度检查
          (1) 语义相关性：answer和question的embedding余弦相似度
          (2) 问题意图匹配：答案是否包含问题的核心疑问词和关键实体

        两者的几何平均确保：
        - 答案既在语义上和问题相关(不跑题)
        - 又确实在尝试回答用户问的问题(不答非所问)
        """
        if not answer or not question:
            return 0.0

        # (1) 语义相关性
        ans_vec = self._embed_text(answer)
        q_vec = self._embed_text(question)
        semantic_rel = self._cosine_similarity(ans_vec, q_vec)

        # (2) 问题意图匹配：提取问题中的核心实体和疑问
        q_keywords = self._tokenize_keywords(question)
        a_keywords = self._tokenize_keywords(answer)
        if not q_keywords:
            return semantic_rel

        # 核心实体覆盖：问题中提到的名词/关键词在答案中出现的比例
        entity_coverage = len(q_keywords & a_keywords) / len(q_keywords)

        # 几何平均：两个维度都必须高才算真正相关
        relevance = (semantic_rel * entity_coverage) ** 0.5
        return min(1.0, relevance)


class RAGEvaluator:
    """RAG系统评估器"""
    def __init__(self, rag_system):
        self.rag_system = rag_system
        # [OPT-04] 使用企业级Judge替换SimpleJudge，传入模型引用
        embeddings = rag_system.embeddings if hasattr(rag_system, 'embeddings') else None
        rerank_model = rag_system.rerank_model if hasattr(rag_system, 'rerank_model') else None
        self.judge = EnterpriseJudge(embeddings=embeddings, rerank_model=rerank_model)
        self.results: List[EvalResult] = []

    def load_testset(self, csv_path: str) -> pd.DataFrame:
        """加载测试集并进行数据质量检查"""
        possible_paths = [
            csv_path,
            f"./{csv_path}",
            f"../{csv_path}",
            f"../../{csv_path}",
            f"./RAG_files/{csv_path}",
        ]
        found = csv_path
        for path in possible_paths:
            if os.path.exists(path):
                found = path
                break
        else:
            raise FileNotFoundError(f"找不到测试集文件: {csv_path}")

        df = pd.read_csv(found, encoding='utf-8')
        # 数据质量检查
        required_cols = ['question', 'ground_truth']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"测试集缺少必要列: {missing}")

        print(f"加载测试集: {os.path.basename(found)}, 共{len(df)}条")
        return df

    @retry_on_failure(max_retries=1, delay=2.0)
    def _retrieve_safe(self, question: str) -> List:
        """带重试保护的检索调用"""
        return self.rag_system._retrieve_with_rerank(question)

    @retry_on_failure(max_retries=1, delay=2.0)
    def _generate_safe(self, question: str, contexts: list = None):
        """带重试保护的生成调用；传入contexts时跳过重复检索"""
        if self.rag_system.rag_chain is None:
            self.rag_system._build_rag_chain()
        if contexts is not None:
            return self.rag_system._generate_answer(question, contexts)
        return self.rag_system.rag_chain.invoke({"input": question})

    def evaluate_single(self, row: pd.Series) -> EvalResult:
        """评估单个问题"""
        question = str(row.get('question', '') or '')
        ground_truth = str(row.get('ground_truth', '') or '')
        question_type = str(row.get('question_type', 'unknown') or 'unknown')
        difficulty = str(row.get('difficulty', 'unknown') or 'unknown')

        result = EvalResult(
            question=question,
            ground_truth=ground_truth,
            retrieved_answer='',
            retrieved_contexts=[],
            question_type=question_type,
            difficulty=difficulty,
        )

        total_start = time.time()
        retrieval_start = time.time()

        try:
            # 检索阶段
            contexts = self._retrieve_safe(question)
            result.retrieved_contexts = [
                doc.page_content if hasattr(doc, 'page_content') else str(doc)
                for doc in (contexts or [])
            ]
            result.retrieved_count = len(result.retrieved_contexts)
            result.retrieval_time_ms = (time.time() - retrieval_start) * 1000

            # 检查检索相关性
            if ground_truth and len(ground_truth) > 5:
                all_context = " ".join(result.retrieved_contexts)
                gt_words = self.judge._tokenize_keywords(ground_truth)
                ctx_words = self.judge._tokenize_keywords(all_context)
                if gt_words:
                    overlap = len(gt_words & ctx_words) / len(gt_words)
                    result.has_relevant_context = overlap > 0.15

            # 生成阶段（复用已检索context，避免rag_chain内二次检索）
            generation_start = time.time()
            response = self._generate_safe(question, contexts=contexts)
            result.retrieved_answer = response if isinstance(response, str) else str(response)
            result.generation_time_ms = (time.time() - generation_start) * 1000
            result.total_time_ms = (time.time() - total_start) * 1000

        except Exception as e:
            result.total_time_ms = (time.time() - total_start) * 1000
            return result

        # [OPT-04] 企业级指标计算
        if ground_truth and len(ground_truth) > 0:
            try:
                result.answer_accuracy = self.judge.compute_answer_accuracy(
                    result.retrieved_answer, ground_truth
                )
                result.answer_relevance = self.judge.compute_answer_relevance(
                    result.retrieved_answer, question
                )
                result.context_recall = self.judge.compute_context_recall(
                    ground_truth, result.retrieved_contexts
                )
                result.faithfulness = self.judge.compute_faithfulness(
                    question, result.retrieved_answer, result.retrieved_contexts
                )

                # 子维度指标
                if result.faithfulness is not None:
                    result.hallucination_rate = max(0.0, 1.0 - result.faithfulness)

            except Exception:
                pass

        return result

    def run_evaluation(self, testset_path: str, limit: int = None,
                       save_results: bool = True) -> dict:
        """运行完整评估"""
        df = self.load_testset(testset_path)
        valid_df = df[
            df['ground_truth'].notna() &
            (df['ground_truth'].astype(str).str.strip() != '') &
            (df['question'].notna()) &
            (df['question'].astype(str).str.strip() != '')
        ]
        print(f"有效测试用例: {len(valid_df)}/{len(df)}")

        if limit:
            valid_df = valid_df.head(limit)
            print(f"限制评估数量: {limit}")

        type_counts = valid_df['question_type'].value_counts()
        print("测试用例分布:")
        for qtype, count in type_counts.items():
            print(f"   {qtype}: {count}")

        print("开始评估...")
        self.results = []

        for idx, (_, row) in enumerate(tqdm(
            valid_df.iterrows(), total=len(valid_df), desc="评估进度"
        )):
            result = self.evaluate_single(row)
            self.results.append(result)

            if (idx + 1) % 5 == 0 and self.results:
                recent = self.results[-5:]
                avg_acc = np.mean([r.answer_accuracy for r in recent])
                avg_faith = np.mean([r.faithfulness for r in recent])

        summary = self._compute_summary()
        self._print_summary(summary)

        if save_results:
            self._save_results()

        return summary

    def _compute_summary(self) -> dict:
        """计算汇总指标 - 扩展维度"""
        if not self.results:
            return {}

        def safe_mean(values):
            valid = [v for v in values if v is not None]
            return float(np.mean(valid)) if valid else 0.0

        metrics = {
            'total_questions': len(self.results),
            'avg_answer_accuracy': safe_mean([r.answer_accuracy for r in self.results]),
            'avg_answer_relevance': safe_mean([r.answer_relevance for r in self.results]),
            'avg_faithfulness': safe_mean([r.faithfulness for r in self.results]),
            'avg_context_recall': safe_mean([r.context_recall for r in self.results]),
            'avg_hallucination_rate': safe_mean([r.hallucination_rate for r in self.results]),
            'avg_retrieval_time_ms': safe_mean([r.retrieval_time_ms for r in self.results]),
            'avg_generation_time_ms': safe_mean([r.generation_time_ms for r in self.results]),
            'avg_total_time_ms': safe_mean([r.total_time_ms for r in self.results]),
            'avg_retrieved_count': safe_mean([r.retrieved_count for r in self.results]),
            'has_relevant_context_rate': safe_mean([
                1 if r.has_relevant_context else 0 for r in self.results
            ]),
            'success_rate': safe_mean([
                1 if r.answer_accuracy > 0.3 else 0 for r in self.results
            ]),
            'excellent_rate': safe_mean([
                1 if r.answer_accuracy > 0.6 else 0 for r in self.results
            ]),
        }

        # 按难度分组
        difficulties = set(r.difficulty for r in self.results if r.difficulty)
        metrics['by_difficulty'] = {}
        for diff in difficulties:
            dr = [r for r in self.results if r.difficulty == diff]
            if dr:
                metrics['by_difficulty'][diff] = {
                    'count': len(dr),
                    'avg_accuracy': safe_mean([r.answer_accuracy for r in dr]),
                    'avg_faithfulness': safe_mean([r.faithfulness for r in dr]),
                    'avg_recall': safe_mean([r.context_recall for r in dr]),
                }

        # 按问题类型分组
        question_types = set(r.question_type for r in self.results if r.question_type)
        metrics['by_question_type'] = {}
        for qtype in question_types:
            tr = [r for r in self.results if r.question_type == qtype]
            if tr:
                metrics['by_question_type'][qtype] = {
                    'count': len(tr),
                    'avg_accuracy': safe_mean([r.answer_accuracy for r in tr]),
                    'avg_faithfulness': safe_mean([r.faithfulness for r in tr]),
                    'avg_recall': safe_mean([r.context_recall for r in tr]),
                }

        return metrics

    def _print_summary(self, summary: dict):
        """打印汇总报告"""
        if not summary:
            return
        print("\n" + "=" * 60)
        print("  评估结果汇总")
        print("=" * 60)
        print(f"  总问题数:           {summary.get('total_questions', 0)}")
        print(f"  答案准确性(avg):    {summary.get('avg_answer_accuracy', 0):.4f}")
        print(f"  答案相关性(avg):    {summary.get('avg_answer_relevance', 0):.4f}")
        print(f"  忠实度(avg):        {summary.get('avg_faithfulness', 0):.4f}")
        print(f"  上下文召回率(avg):  {summary.get('avg_context_recall', 0):.4f}")
        print(f"  幻觉率(avg):        {summary.get('avg_hallucination_rate', 0):.4f}")
        print(f"  相关上下文命中率:   {summary.get('has_relevant_context_rate', 0):.2%}")
        print(f"  成功率(acc>0.3):    {summary.get('success_rate', 0):.2%}")
        print(f"  优秀率(acc>0.6):    {summary.get('excellent_rate', 0):.2%}")
        print(f"  平均检索耗时:       {summary.get('avg_retrieval_time_ms', 0):.0f}ms")
        print(f"  平均生成耗时:       {summary.get('avg_generation_time_ms', 0):.0f}ms")
        print(f"  平均总耗时:         {summary.get('avg_total_time_ms', 0):.0f}ms")
        print("=" * 60)

    def _save_results(self):
        """保存评估结果"""
        os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_df = pd.DataFrame([r.to_dict() for r in self.results])
        results_path = os.path.join(EVAL_RESULTS_DIR, f"eval_results_{timestamp}.csv")
        results_df.to_csv(results_path, index=False, encoding='utf-8')
        print(f"\n详细结果: {results_path}")

        summary = self._compute_summary()
        summary_path = os.path.join(EVAL_RESULTS_DIR, f"eval_summary_{timestamp}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"汇总指标: {summary_path}")
