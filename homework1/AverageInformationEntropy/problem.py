import jieba
import math
from opencc import OpenCC
from collections import Counter


# 没有出现在停词表的标点符号或内容，可能影响分词和内容准确性
useless = ["\n", "　", " ", "本书来自www.cr173.com免费txt小说下载站", "更多更新免费电子书请关注www.cr173.com"]

# 删除无用内容
def erase_useless(book):
    for word in useless:
        book = book.replace(word, "")
    return book

# 获取中文停词表
def get_cn_stopwords(path):
    cn_stopwords = open(path, "r").read()
    cn_stopwords_list = cn_stopwords.split("\n")
    cn_stopwords_list = list(filter(None, cn_stopwords_list))
    return set(cn_stopwords_list)

# 取出所有相邻的n个元素
def n_gram(words, n):
    n_g = [words[i:i+n] for i in range(len(words)-n+1)]
    return n_g

if __name__ == "__main__":
    cn_stopwords_set = get_cn_stopwords("./cn_stopwords.txt") # 获取中文停词表
    word_dict = dict() # 初始化字典
    cc = OpenCC("t2s") # 初始化opencc，繁体转简体
    corpus_path = "./chinese-corpus" # 语料库路径
    corpus_list = open("{}/inf.txt".format(corpus_path), "r", encoding="gb18030").readline().split(",") # 读取语料库目录为list
    content = "" # 保存所有字
    words = [] # 保存所有单词
    # 对于目录中每一个书
    for title in corpus_list:
        book = open("{}/{}.txt".format(corpus_path, title), "r", encoding="gb18030").read() # 按对应编码格式打开文件并读取内容
        book = erase_useless(book) # 删除无用内容
        book = cc.convert(book) # 繁体转简体
        content = content + book # 将内容加入到字符串中
        seg_list = jieba.cut(book, cut_all=False) # 分词
        for word in seg_list:
            # 若词不在停词表中
            if word not in cn_stopwords_set:
                words.append(word)

    ######按字计算平均信息熵######
    # 一元模型信息熵计算
    content_gram_1 = n_gram(content, 1)
    content_freq_1 = Counter(content_gram_1)
    content_length_1 = len(content_gram_1)
    content_entropy_list_1 = []
    for v, count in content_freq_1.items():
        p = count / content_length_1
        content_entropy_list_1.append(p * math.log2(p))
    content_entorpy_1 = -1.0 * sum(content_entropy_list_1)
    print("按字计算的一元信息熵为: {} bit/字".format(content_entorpy_1))
    # 二元模型信息熵计算
    content_gram_2 = n_gram(content, 2)
    content_freq_2 = Counter(content_gram_2)
    content_length_2 = len(content_gram_2)
    content_entropy_list_2 = []
    for v, count in content_freq_2.items():
        p_x_and_y = count / content_length_2
        p_x_when_y = count / content_freq_1[v[0]]
        content_entropy_list_2.append(p_x_and_y * math.log2(p_x_when_y))
    content_entorpy_2 = -1.0 * sum(content_entropy_list_2)
    print("按字计算的二元信息熵为: {} bit/字".format(content_entorpy_2))
    # 三元模型信息熵计算
    content_gram_3 = n_gram(content, 3)
    content_freq_3 = Counter(content_gram_3)
    content_length_3 = len(content_gram_3)
    content_entropy_list_3 = []
    for v, count in content_freq_3.items():
        p_x_and_y_and_z = count / content_length_3
        p_x_when_yz = count / content_freq_2[v[0:2]]
        content_entropy_list_3.append(p_x_and_y_and_z * math.log2(p_x_when_yz))
    content_entorpy_3 = -1.0 * sum(content_entropy_list_3)
    print("按字计算的三元信息熵为: {} bit/字".format(content_entorpy_3))


    ######按词计算平均信息熵######
    # 一元模型信息熵计算
    words_freq_1 = Counter(words)
    words_length_1 = len(words)
    words_entropy_list_1 = []
    for v, count in words_freq_1.items():
        p = count / words_length_1
        words_entropy_list_1.append(p * math.log(p, 2))
    words_entropy_1 = -1.0 * sum(words_entropy_list_1)
    print("按词计算的一元信息熵为: {} bit/词".format(words_entropy_1))
    # 二元模型信息熵计算
    words_gram_2 = n_gram(words, 2) # 拆分后为list的嵌套
    words_gram_2 = [" ".join(w) for w in words_gram_2] # 使用空格连接两个词，并且后续能够进行拆分
    words_freq_2 = Counter(words_gram_2)
    words_length_2 = len(words_gram_2)
    words_entropy_list_2 = []
    for v, count in words_freq_2.items():
        p_x_and_y = count / words_length_2
        p_x_when_y = count / words_freq_1[v.split(" ")[0]]
        words_entropy_list_2.append(p_x_and_y * math.log2(p_x_when_y))
    words_entropy_2 = -1.0 * sum(words_entropy_list_2)
    print("按词计算的二元信息熵为: {} bit/词".format(words_entropy_2))
    # 三元模型信息熵计算
    words_gram_3 = n_gram(words, 3) # 拆分后为list的嵌套
    words_gram_3 = [" ".join(w) for w in words_gram_3]
    words_freq_3 = Counter(words_gram_3)
    words_length_3 = len(words_gram_3)
    words_entropy_list_3 = []
    for v, count in words_freq_3.items():
        p_x_and_y_and_z = count / words_length_3
        p_x_when_yz = count / words_freq_2[" ".join(v.split(" ")[0:2])]
        words_entropy_list_3.append(p_x_and_y_and_z * math.log2(p_x_when_yz))
    words_entropy_3 = -1.0 * sum(words_entropy_list_3)
    print("按词计算的三元信息熵为: {} bit/词".format(words_entropy_3))