import jieba
import math
from opencc import OpenCC
import matplotlib.pyplot as plt
import pyceres
import numpy as np

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

# 根据词典绘制双对数坐标曲线
def show_result(freq, freq1, freq2, rank):
    a, = plt.loglog(rank, freq) # 使用双对数坐标
    b, = plt.loglog(rank, freq1)
    c, = plt.loglog(rank, freq2)
    plt.legend([a, b, c], ["origin", "k/(r^alpha)", "k/[(r+beta)^alpha]"])
    plt.title("word frequency vs rank")
    plt.xlabel("rank")
    plt.ylabel("word frequency")
    plt.show()

# 按对数坐标系等间隔取样
def get_sequence(rank):
    sequence = np.linspace(0, np.log(rank[-1]), 10000)
    sequence = np.exp(sequence).astype(int)
    return sequence

# 优化损失函数
class k_alpha_CostFunction(pyceres.CostFunction):
    def __init__(self, freq, rank):
        pyceres.CostFunction.__init__(self)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2])
        self.freq = freq
        self.rank = rank

    def Evaluate(self, parameters, residuals, jacobians):
        k = parameters[0][0]
        alpha = parameters[0][1]
        residuals[0] = math.log(self.freq) - math.log(k) + alpha * math.log(self.rank) # 残差计算
        # 雅可比矩阵计算，详见报告
        if jacobians is not None:
            jacobians[0][0] = -1.0 / k
            jacobians[0][1] = math.log(self.rank)
        return True

# 拟合公式(5)，只拟合k和alpha
def optimization_k_alpha(freq, rank):
    problem = pyceres.Problem() # 创建优化问题
    param = np.array([1.0, 1.0]) # k与alpha的初值
    seq = get_sequence(rank) # 不是将所有观测值加入，而是按照对数坐标等间隔取样
    for s in seq:
        if s == 0:
            continue
        cost = k_alpha_CostFunction(freq[s], s) # 创建损失函数
        problem.add_residual_block(cost, None, [param]) # 添加残差块
    options = pyceres.SolverOptions() # 求解选项
    summary = pyceres.SolverSummary() # 求解统计信息
    pyceres.solve(options, problem, summary) # 求解问题
    plt.loglog(rank, freq)
    k = param[0]
    alpha = param[1]
    print("k = {}, alpha = {}".format(k, alpha))
    return k / np.power(rank, alpha)

# 优化损失函数
class k_alpha_beta_CostFunction(pyceres.CostFunction):
    def __init__(self, freq, rank):
        pyceres.CostFunction.__init__(self)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([3])
        self.freq = freq
        self.rank = rank

    def Evaluate(self, parameters, residuals, jacobians):
        k = parameters[0][0]
        alpha = parameters[0][1]
        beta = parameters[0][2]
        # 防止对小于0的数进行log操作
        if self.rank + beta <= 0:
            return False
        residuals[0] = math.log(self.freq) - math.log(k) + alpha * math.log(self.rank + beta) # 残差计算
        # 雅可比矩阵计算，详见报告
        if jacobians is not None:
            jacobians[0][0] = -1.0 / k
            jacobians[0][1] = math.log(self.rank + beta)
            jacobians[0][2] = alpha / (self.rank + beta)
        return True

# 拟合公式(6)，同时拟合k、alpha、beta
def optimization_k_alpha_beta(freq, rank):
    problem = pyceres.Problem() # 创建优化问题
    param = np.array([1.0, 1.0, 1.0]) # k、alpha、beta的初值
    seq = get_sequence(rank) # 不是将所有观测值加入，而是按照对数坐标等间隔取样
    for s in seq:
        if s == 0:
            continue
        cost = k_alpha_beta_CostFunction(freq[s], s) # 创建损失函数
        problem.add_residual_block(cost, None, [param]) # 添加残差块
    options = pyceres.SolverOptions() # 求解选项
    summary = pyceres.SolverSummary() # 求解统计信息
    pyceres.solve(options, problem, summary) # 求解问题
    plt.loglog(rank, freq)
    k = param[0]
    alpha = param[1]
    beta = param[2]
    print("k = {}, alpha = {}, beta = {}".format(k, alpha, beta))
    return k / np.power(rank + beta, alpha)


if __name__ == "__main__":
    cn_stopwords_set = get_cn_stopwords("./cn_stopwords.txt") # 获取中文停词表
    word_dict = dict() # 初始化字典
    cc = OpenCC("t2s") # 初始化opencc，繁体转简体
    corpus_path = "./chinese-corpus" # 语料库路径
    corpus_list = open("{}/inf.txt".format(corpus_path), "r", encoding="gb18030").readline().split(",") # 读取语料库目录为list
    # 对于目录中每一个书
    for title in corpus_list:
        book = open("{}/{}.txt".format(corpus_path, title), "r", encoding="gb18030").read() # 按对应编码格式打开文件并读取内容
        book = erase_useless(book) # 删除无用内容
        book = cc.convert(book) # 繁体转简体
        seg_list = jieba.cut(book, cut_all=False) # 分词
        for word in seg_list:
            # 若词不在停词表中
            if word not in cn_stopwords_set:
                word_dict[word] = word_dict.get(word, 0) + 1 # 字典中对应的值+1
    freq = [v for v in sorted(word_dict.values(),reverse=True)] # 对词频进行排序
    rank = [i for i in range(0, len(freq))] # 词频排名
    freq1 = optimization_k_alpha(freq, rank) # 拟合参数k、alpha
    freq2 = optimization_k_alpha_beta(freq, rank) # 拟合参数k、alpha、beta
    show_result(freq, freq1, freq2, rank) # 结果可视化
