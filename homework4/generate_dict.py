import jieba
import opencc
import gensim.corpora
import gensim.models

import utility

def main():
    cc = opencc.OpenCC("t2s") # 初始化opencc，繁体转简体
    corpus_path = "./chinese-corpus" # 语料库路径
    title_list = open("{}/inf.txt".format(corpus_path), "r", encoding="gb18030").readline().split(",") # 读取语料库目录为list
    corpus_tokenized = list()
    # 对于目录中每一个书
    for title in title_list:
        book = open("{}/{}.txt".format(corpus_path, title), "r", encoding="gb18030").read() # 按对应编码格式打开文件并读取内容
        book = utility.erase_useless(book) # 删除无用内容
        book = utility.preprocessing(book) # 预处理
        book = cc.convert(book) # 繁体转简体
        sentences = utility.split_sentences(book) # 将文章拆分为单独句子，格式为list[str]
        # 对每一个句子进行分词，分词结果保存在list[list[str]]，
        # 外层list保存句子，内层list为每个句子分词后的结果
        # 要求词包含中文字符
        sentences_tokenized = [[word for word in jieba.cut(sentence, cut_all=False) if utility.is_chinese_token(word)] for sentence in sentences]
        sentences_tokenized = [s for s in sentences_tokenized if len(s) != 0] # 去除空句子（可能整句都在停词表中）
        corpus_tokenized.extend(sentences_tokenized) # 加入到总的list中

    dictionary = gensim.corpora.Dictionary([["<bos>", "<eos>", "<pad>"]]) # 创建字典，加入<bos><eos><pad>
    dictionary.add_documents(corpus_tokenized) # 将预料库加入字典
    dictionary.save("./result/dictionary.bin") # 保存字典

if __name__ == "__main__":
    main()