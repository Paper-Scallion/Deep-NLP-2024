import re
import jieba
import opencc
import gensim.corpora
import numpy as np

import mittens

import utility

# 获取共现矩阵
def get_cooccurrence(dictionary, corpus_tokenized, window):
    cooccurrence = np.zeros([len(dictionary), len(dictionary)], dtype=np.float32)
    for sentences_tokenized in corpus_tokenized: # 遍历每个句子
        for i in range(len(sentences_tokenized)): # 遍历每个句子中的词
            current_word = sentences_tokenized[i] # 当前词
            try:
                current_id = dictionary.token2id[current_word] # 当前id
            except:
                continue # 该词不在字典中，跳过
            for j in list(range(-window, 0)) + list(range(1, window + 1)): # 在window范围内
                if i + j >= 0 and i + j < len(sentences_tokenized): # 防止超出索引范围
                    neighbor_word = sentences_tokenized[i + j] # 滑窗内的词
                    try:
                        neighbor_id = dictionary.token2id[neighbor_word] # 相邻词id
                    except:
                        continue
                    cooccurrence[current_id, neighbor_id] += 1 # 共现矩阵对应位置值+1
    return cooccurrence

def main():
    cn_stopwords_set = utility.get_cn_stopwords("./cn_stopwords.txt") # 获取中文停词表
    cc = opencc.OpenCC("t2s") # 初始化opencc，繁体转简体
    corpus_path = "./chinese-corpus" # 语料库路径
    title_list = open("{}/inf.txt".format(corpus_path), "r", encoding="gb18030").readline().split(",") # 读取语料库目录为list
    corpus_tokenized = list()
    # 对于目录中每一个书
    for title in title_list:
        book = open("{}/{}.txt".format(corpus_path, title), "r", encoding="gb18030").read() # 按对应编码格式打开文件并读取内容
        book = utility.erase_useless(book) # 删除无用内容
        book = cc.convert(book) # 繁体转简体
        sentences = re.split(r"!|\。|\?", book) # 将文章拆分为单独句子，格式为list[str]
        # 对每一个句子进行分词，分词结果保存在list[list[str]]，
        # 外层list保存句子，内层list为每个句子分词后的结果
        # 同时去除了停词，并且要求词包含中文字符
        sentences_tokenized = [[word for word in jieba.cut(sentence, cut_all=False) if word not in cn_stopwords_set and utility.contains_chinese_characters(word)] for sentence in sentences]
        sentences_tokenized = [s for s in sentences_tokenized if len(s) != 0] # 去除空句子（可能整句都在停词表中）
        corpus_tokenized.extend(sentences_tokenized) # 加入到总的list中

    dictionary = gensim.corpora.Dictionary(corpus_tokenized) # 创建字典
    dictionary.save("./result/GloVe/dictionary.bin") # 保存字典

    dictionary.filter_extremes(no_below=25) # 获取出现频次>=25的高频词
    dictionary.save("./result/GloVe/filtered_dictionary.bin") # 保存字典

    cooccurrence = get_cooccurrence(dictionary, corpus_tokenized, 5) # 计算共现矩阵
    np.save("./result/GloVe/cooccurrence.npy", cooccurrence) # 保存共现矩阵

    model = mittens.GloVe(n=64, max_iter=1000)
    embeddings = model.fit(cooccurrence) # 训练GloVe模型
    np.save("./result/GloVe/embeddings.npy", embeddings) # 保存embedding矩阵

if __name__ == "__main__":
    main()