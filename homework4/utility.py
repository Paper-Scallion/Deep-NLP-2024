import re
import opencc
import jieba
import numpy as np

# 没有出现在停词表的标点符号或内容，可能影响分词和内容准确性
useless = [
    "\n",
    "　",
    " ",
    "本书来自www.cr173.com免费txt小说下载站",
    "更多更新免费电子书请关注www.cr173.com",
]


# 获取中文停词表
def get_cn_stopwords(path):
    cn_stopwords = open(path, "r").read()
    cn_stopwords_list = cn_stopwords.split("\n")
    cn_stopwords_list = list(filter(None, cn_stopwords_list))
    return set(cn_stopwords_list)


# 检查字符串s中是否包含中文汉字
def contains_chinese_characters(s):
    chinese_pattern = re.compile(r"[\u4e00-\u9fff]+")
    return bool(chinese_pattern.search(s))


def is_chinese_token(s):
    if len(s) == 1 and s[0] in ["“", "”", "：", "，", "。", "？", "！", "（", "）", "…", "、", "；"]:
        return True
    for c in s:
        if not "\u4e00" <= c <= "\u9fff":
            return False
    return True


# 删除无用内容
def erase_useless(book):
    for word in useless:
        book = book.replace(word, "")
    return book


# 预处理
def preprocessing(book: str):
    # 统一符号
    replace_context = [
        ["「", "“"],
        ["」", "”"],
        ["『", "“"],
        ["』", "”"],
        [":", "："],
        [",", "，"],
        [".", "。"],
        ["?", "？"],
        ["!", "！"],
        ["(", "（"],
        [")", "）"],

    ]
    for replace in replace_context:
        book = book.replace(replace[0], replace[1])
    # 统一省略号
    book = re.sub(r"…+", "…", book)
    return book

def split_sentences(text):
    parts = re.split('([。！？…])', text)
    # 过滤掉空字符串，并确保每个有效部分末尾保留句号、叹号或问号
    cleaned_parts = [part + punctuation for part, punctuation in zip(parts[::2], parts[1::2])]
    # 如果原文以句号、叹号、问号或省略号结尾，最后一个元素会是多余的标点，需要去掉
    if cleaned_parts and cleaned_parts[-1] in '。！？…':
        cleaned_parts.pop()
    return cleaned_parts

# 获取段落
def get_paragraph(paragraph_path):
    paragraph = open(paragraph_path, "r").read()
    cc = opencc.OpenCC("t2s")
    paragraph = cc.convert(paragraph)
    cn_stopwords_set = get_cn_stopwords("./cn_stopwords.txt")
    paragraph_tokenized = [
        word
        for word in jieba.cut(paragraph, cut_all=False)
        if word not in cn_stopwords_set and contains_chinese_characters(word)
    ]
    return paragraph_tokenized


# 获取段落向量
def get_paragraph_vector_Word2Vec(model, paragraph_tokenized):
    word_vectors = np.array(
        [model.wv[word] for word in paragraph_tokenized]
    )  # 提取词向量
    paragraph_vector = np.mean(word_vectors, axis=0)  # 获取平均词向量
    return paragraph_vector


# 判断token是否在词典中
def in_dictionary(dictionary, token):
    ret = False
    try:
        dictionary.token2id[token]
        ret = True
    except:
        ret = False
    return ret


# 获取段落向量
def get_paragraph_vector_GloVe(dictionary, embeddings, paragraph_tokenized):
    word_vectors = np.array(
        [
            embeddings[dictionary.token2id[token]]
            for token in paragraph_tokenized
            if in_dictionary(dictionary, token)
        ]
    )  # 提取词向量
    paragraph_vector = np.mean(word_vectors, axis=0)  # 获取平均词向量
    return paragraph_vector


# 余弦相似度计算
def cosine_distance(vector_1, vector_2):
    distance = np.dot(vector_1, vector_2) / (
        np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
    )
    return distance


# 余弦相似度计算
def cosine_similarities(vector_1, vectors_all):
    norm = np.linalg.norm(vector_1)
    all_norms = np.linalg.norm(vectors_all, axis=1)
    dot_products = np.dot(vectors_all, vector_1)
    similarities = dot_products / (norm * all_norms)
    return similarities


# 查找与目标词余弦相似度最高的n个词
def most_similar(dictionary, embeddings, word, n):
    word_id = dictionary.token2id[word]  # 从词典获取id
    word_vector = embeddings[word_id]  # 从embeddings获取词向量
    similarities = cosine_similarities(word_vector, embeddings)  # 获取余弦相似度矩阵
    sorted_indices = np.argsort(similarities)[::-1]  # 获取索引
    most_similar_list = [
        (dictionary[id], similarities[id]) for id in sorted_indices[1 : n + 1]
    ]  # 获取除自身外相似度最高的前10个词及其余弦相似度
    return most_similar_list
