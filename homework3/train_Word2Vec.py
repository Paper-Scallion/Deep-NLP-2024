import re
import jieba
import opencc
import gensim.corpora
import gensim.models

import utility

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
    dictionary.save("./result/Word2Vec/dictionary.bin") # 保存字典

    model = gensim.models.Word2Vec(sentences=corpus_tokenized, vector_size=64, window=5, min_count=1, workers=16, epochs=1000) # 训练Word2Vec模型
    model.save("./result/Word2Vec/model.bin") # 保存模型

if __name__ == "__main__":
    main()