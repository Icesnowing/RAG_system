"""
查询预处理模块
对应原 rag_system_code.py 中 LocalRAGSystem 的：
    _preprocess_query / _fix_full_half_width
    _extract_keywords_llm / _extract_keywords_rule_based / _extract_keywords / _augment_query
以 Mixin 形式提供给主系统类使用，方法实现与原文件保持一致。
"""

import re

import jieba
#导入对话提示词模板
from langchain_core.prompts import ChatPromptTemplate

from config import MAX_KEYWORDS, USE_LLM_FOR_KEYWORDS, ENABLE_KEYWORD_AUGMENT


class QueryPreprocessMixin:
    """查询清洗、全半角修复、关键词提取与增强能力（供 LocalRAGSystem 多继承组合）"""

    #===============query预处理=====================
    def _preprocess_query(self, query: str) -> str:
        """
        [OPT-02] 优化的Query预处理。

        原版问题：
        - 删除所有中文标点并把词拼在一起，破坏语义结构
        - 全角→半角转换方向错误
        - 停用词过滤后去掉所有空格，导致英文短语丢失边界

        优化策略：
        1. 保留核心标点和空格作为词语边界
        2. 仅过滤真正的停用词（保留技术术语）
        3. 保持中英文混排的自然结构
        4. 保留数字、单位、缩写
        """
        if not isinstance(query, str) or not query.strip():
            return ""

        original_query = query
        query = query.strip()

        # [OPT-02] 全角→半角转换（修复方向）
        query = self._fix_full_half_width(query)

        # 去除换行/制表符，保留空格
        query = query.replace("\n", " ").replace("\r", " ").replace("\t", " ")

        # [OPT-02] 仅去除极特殊字符，保留中英文标点和数字
        query = re.sub(r'[^一-龥a-zA-Z0-9，。！？；：""''（）【】.,!?[]()-+=/%#&*]', ' ', query)

        # 去除常见问候语前缀
        greetings = [
            '你好', '您好', '请问', '我想问', '能不能', '可以告诉我', '麻烦问一下',
            '帮忙看看', '请教一下', '咨询一下', '了解一下', '想知道', '请问一下',
        ]
        for greeting in greetings:
            if query.lower().startswith(greeting.lower()):
                query = query[len(greeting):]
                break

        # 合并多余空格
        query = re.sub(r'\s{2,}', ' ', query).strip()

        # 如果清洗后为空或过短，返回原始查询
        if len(query.replace(' ', '')) < 2:
            return original_query.strip()

        return query

    def _fix_full_half_width(self, text: str) -> str:
        """
        [OPT-03] 修复：正确的全角→半角转换。

        原版bug：将半角空格(32)转全角空格(12288)，方向完全反了。
        修复：全角字符(FF01-FF5E)转半角(21-7E)，全角空格(3000)转半角空格(32)。
        """
        result = []
        for char in text:
            code = ord(char)
            if 0xFF01 <= code <= 0xFF5E:
                code -= 0xFEE0
            elif code == 0x3000:  # 全角空格
                code = 0x0020
            result.append(chr(code))
        return "".join(result)

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
                word_lower not in self._stopwords and
                flag in allowed_pos):
                candidate_words.append(word_lower)
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1

        # 3. 如果没有有效的候选词，放宽条件：保留所有长度>=2且不是纯停用词的词
        if not candidate_words:
            words = jieba.cut(query)
            for word in words:
                word_lower = word.lower().strip()
                if len(word_lower) >= 2 and word_lower not in self._stopwords:
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
