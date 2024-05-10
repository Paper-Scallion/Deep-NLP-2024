import random

import jieba
from opencc import OpenCC
import numpy as np

import tqdm

# 参数
paragraph_number = 1000  # 抽取的段落数
sample_len = 3000  # 一个段落抽取数量
token_is_word = True  # 设置单词为基本单元
useless = ["\n", "　", " ", "本书来自www.cr173.com免费txt小说下载站", "更多更新免费电子书请关注www.cr173.com"]  # 没有出现在停词表的标点符号或内容，可能影响分词和内容准确性

K = 100  # 主题数

# 全局变量，由程序自动配置
train_M = 16  # 文章数
eval_M = paragraph_number  # 评估的段落数

train_len = round(sample_len * 0.9)  # 一个段落作为训练数据的token数
eval_len = sample_len - train_len  # 一个段落作为评估数据的token数

V = 0  # 训练语料的词表单词数

train_token_length = 0  # 每篇文章具有的最长token数
eval_token_length = 0

token_map = dict()


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


# 从语料库获取全部token和label，并对每一个token生成token_id
def data_preprocess(path):
    token_list = []  # token集合
    label_list = []  # 标签
    label = 0  # 当前标签
    cn_stopwords_set = get_cn_stopwords("./cn_stopwords.txt")  # 获取中文停词表
    cc = OpenCC("t2s")  # 初始化opencc，繁体转简体
    corpus_list = open("{}/inf.txt".format(path), "r", encoding="gb18030").readline().split(",")  # 读取语料库目录为list
    global train_M
    train_M = len(corpus_list)
    for title in corpus_list:  # 对于目录中每一个书
        book = open("{}/{}.txt".format(path, title), "r", encoding="gb18030").read()  # 按对应编码格式打开文件并读取内容
        book = erase_useless(book)  # 删除无用内容
        book = cc.convert(book)  # 繁体转简体
        seg_list = jieba.cut(book, cut_all=False)  # 分词
        for word in seg_list:
            # 若词不在停词表中
            if word not in cn_stopwords_set:
                if token_is_word:  # 使用单词作为基本单元
                    token_list.append(word)  # 记录token
                    label_list.append(label)  # 记录label
                else:  # 使用字作为基本单元
                    lw = list(word)
                    token_list.extend(lw)  # 将词拆分为字再记录
                    label_list.extend([label for _ in range(len(lw))])
        label += 1
    return token_list, label_list


# 从语料库拆分得到训练集和测试集
def split_tokens(token_list, label_list):
    train_set = []  # 训练语料
    eval_set = []  # 评估语料
    eval_gt = []  # 评估真值
    sample_interval = round(len(token_list) / paragraph_number)  # 取样间隔
    sample_list = list(range(sample_len))  # 编号
    for i in range(paragraph_number):  # 抽取段落
        begin = i * sample_interval  # 抽取起始编号
        sample_tokens = token_list[begin:begin + sample_len]  # 抽取一个段落，长度为sample_len
        sample_labels = label_list[begin:begin + sample_len]  # 连带标签一起抽取
        assert (x == sample_labels[0] for x in sample_labels)  # 确保此次抽取均来自同一个txt
        random.shuffle(sample_list)  # 随机打乱
        train_tokens = [sample_tokens[sample_list[i]] for i in range(0, train_len)]  # 打乱后的前一部分为训练数据
        eval_tokens = [sample_tokens[sample_list[i]] for i in range(train_len, sample_len)]  # 打乱后的后一部分为评估数据
        train_set.append((train_tokens, sample_labels[0]))  # 加入到训练集，以元组存储数据和标签
        eval_set.append((eval_tokens, sample_labels[0]))  # 加入到评估集，以元组存储数据和标签
        eval_gt.append(sample_labels[0])
    return train_set, eval_set, eval_gt


# 将数据集重新组合，每个文章的token放在一起
def concat_dataset(token_set):
    new_token_set = [[] for _ in range(train_M)]
    for t, m in token_set:
        new_token_set[m].extend(t)
    return new_token_set


# 根据训练集和测试集生成token_map
def generate_token_map(dataset1, dataset2):
    token_id_dict = dict()
    token_id = 0
    for doc in dataset1:
        for token in doc:
            if token not in token_id_dict:
                token_id_dict[token] = token_id
                token_id += 1
    for doc in dataset2:
        for token in doc:
            if token not in token_id_dict:
                token_id_dict[token] = token_id
                token_id += 1
    return token_id_dict


# 处理评估语料
def handle_eval_dataset(token_set):
    new_token_set = []
    for t, m in token_set:
        new_token_set.append(t)
    return new_token_set


# 从保存概率的数组p中抽取样本
def cumulative_sampling(p):
    for i in range(1, len(p)):
        p[i] += p[i - 1]  # 逐项累积
    u = random.random() * p[-1]  # 随机取值
    for i in range(len(p)):
        if p[i] > u:
            return i


# 计算训练集theta矩阵
def compute_train_theta(nd, ndsum, alpha):
    theta = np.empty([train_M, K], dtype=np.float32)
    for m in range(train_M):
        for k in range(K):
            theta[m, k] = (nd[m, k] + alpha) / (ndsum[m] + K * alpha)
    return theta


# 计算训练集phi矩阵
def compute_train_phi(nw, nwsum, beta):
    phi = np.empty([K, V], dtype=np.float32)
    for k in range(K):
        for v in range(V):
            phi[k, v] = (nw[v, k] + beta) / (nwsum[k] + V * beta)
    return phi


# 计算评估集theta矩阵
def compute_eval_theta(nd, ndsum, alpha):
    theta = np.empty([eval_M, K], dtype=np.float32)
    for m in range(eval_M):
        for k in range(K):
            theta[m, k] = (nd[m, k] + alpha) / (ndsum[m] + K * alpha)
    return theta


# 计算评估集phi矩阵
def compute_eval_phi(train_nw, train_nwsum, eval_nw, eval_nwsum, beta):
    phi = np.empty([K, V], dtype=np.float32)
    for k in range(K):
        for v in range(V):
            w = 0
            phi[k, v] = (train_nw[w, k] + eval_nw[v, k] + beta) / (train_nwsum[k] + eval_nwsum[k] + V * beta)
    return phi


# 保存token_map
def save_model_token_map(token_id_dict):
    f = open('./results/token_map.txt', 'w')
    for key, value in token_id_dict.items():
        f.write("{},{}\n".format(key, value))
    f.close()


# 保存训练语料的主题分布
def save_train_model_tassign(train_set, z):
    f = open('./results/train_tassign.txt', 'w')
    for i in range(train_M):
        for j in range(train_token_length):
            if z[i, j] != -1:
                f.write("{}:{} ".format(token_map[train_set[i][j]], z[i][j]))  # 格式：token_id:主题id
        f.write("\n")
    f.close()


# 保存评估语料的主题分布
def save_eval_model_tassign(eval_set, z):
    f = open('./results/eval_tassign.txt', 'w')
    for i in range(eval_M):
        for j in range(eval_token_length):
            if z[i, j] != -1:
                f.write("{}:{} ".format(token_map[eval_set[i][j]], z[i][j]))  # 格式：token_id:主题id
        f.write("\n")
    f.close()


# 训练模型
def train_model(train_set, alpha, beta, iter_num):
    print("开始训练")
    # 统计量声明
    nw = np.zeros([V, K], dtype=np.int32)  # 主题j对应单词i的实例数
    nwsum = np.zeros(K, dtype=np.int32)  # 主题j对应单词总实例数
    nd = np.zeros([train_M, K], dtype=np.int32)  # 第i篇文章被指定第j个主题词的次数
    ndsum = np.zeros(train_M, dtype=np.int32)  # 第i篇文章词数
    z = -1 * np.ones([train_M, train_token_length], dtype=np.int32)  # 第i篇文章第j个token被指定的topic index
    train_token_map_searched = [[token_map[train_set[m][n]] for n in range(len(train_set[m]))] for m in range(len(train_set))]  # 预先查找token_id，无需每次在dict中查找
    # 初始化
    for m in range(len(train_set)):
        for n in range(len(train_set[m])):
            topic_index = random.randint(0, K - 1)  # 随机指定
            z[m, n] = topic_index  # 记录topic_index
            token_id = train_token_map_searched[m][n]  # 从dict查找token_id
            nw[token_id, topic_index] += 1  # 对应统计量+1
            nwsum[topic_index] += 1
            nd[m, topic_index] += 1
        ndsum[m] = len(train_set[m])
    # 迭代阶段
    for _ in tqdm.tqdm(range(iter_num)):
        for m in range(len(train_set)):
            for n in range(len(train_set[m])):
                t = z[m, n]
                token_id = train_token_map_searched[m][n]
                nw[token_id, t] -= 1
                nwsum[t] -= 1
                nd[m, t] -= 1
                p = np.zeros(K, dtype=np.float32)
                for k in range(K):  # 计算K个主题概率
                    p1 = (nw[token_id, k] + beta) / (nwsum[k] + V * beta)  # 该token在第k个主题的概率
                    p2 = (nd[m, k] + alpha) / (ndsum[m] + K * alpha)  # 第k主题在第m文章中的概率
                    p[k] = p1 * p2
                new_t = cumulative_sampling(p)
                z[m, n] = new_t
                nw[token_id, new_t] += 1
                nwsum[new_t] += 1
                nd[m, new_t] += 1
    # 计算theta与phi
    theta = compute_train_theta(nd, ndsum, alpha)
    phi = compute_train_phi(nw, nwsum, beta)
    # 输出结果
    save_train_model_tassign(train_set, z)  # 保存各文章每个token对应的topic
    np.save("./results/train_theta.npy", theta)  # 保存theta矩阵
    np.save("./results/train_phi.npy", phi)  # 保存phi矩阵
    return nw, nwsum, nd, ndsum, theta, phi


# 预测模型
def predict_model(eval_set, alpha, beta, iter_num, train_nw, train_nwsum):
    print("开始处理验证集")
    # 统计量声明，与训练时对应
    eval_nw = np.zeros([V, K], dtype=np.int32)
    eval_nwsum = np.zeros(K, dtype=np.int32)
    eval_nd = np.zeros([eval_M, K], dtype=np.int32)
    eval_ndsum = np.zeros(eval_M, dtype=np.int32)
    eval_z = -1 * np.ones([eval_M, eval_token_length], dtype=np.int32)
    token_map_searched = [[token_map[eval_set[m][n]] for n in range(len(eval_set[m]))] for m in range(len(eval_set))]
    # 初始化
    for m in range(len(eval_set)):
        for n in range(len(eval_set[m])):
            topic_index = random.randint(0, K - 1)
            eval_z[m, n] = topic_index
            token_id = token_map_searched[m][n]
            eval_nw[token_id, topic_index] += 1
            eval_nwsum[topic_index] += 1
            eval_nd[m, topic_index] += 1
        eval_ndsum[m] = len(eval_set[m])
    # 迭代阶段
    for _ in tqdm.tqdm(range(iter_num)):
        for m in range(len(eval_set)):
            for n in range(len(eval_set[m])):
                t = eval_z[m, n]
                token_id = token_map_searched[m][n]  # 在验证tokenmap中的id
                eval_nw[token_id, t] -= 1
                eval_nwsum[t] -= 1
                eval_nd[m, t] -= 1
                p = np.zeros(K, dtype=np.float32)
                for k in range(K):
                    p1 = (train_nw[token_id, k] + eval_nw[token_id, k] + beta) / (train_nwsum[k] + eval_nwsum[k] + V * beta)
                    p2 = (eval_nd[m, k] + alpha) / (eval_ndsum[m] + K * alpha)
                    p[k] = p1 * p2
                new_t = cumulative_sampling(p)
                eval_nw[token_id, new_t] += 1
                eval_nwsum[new_t] += 1
                eval_nd[m, new_t] += 1
                eval_z[m, n] = new_t
    # 计算theta与phi
    theta = compute_eval_theta(eval_nd, eval_ndsum, alpha)
    phi = compute_eval_phi(train_nw, train_nwsum, eval_nw, eval_nwsum, beta)
    # 输出结果
    save_eval_model_tassign(eval_set, eval_z)
    np.save("./results/eval_theta.npy", theta)  # 保存theta矩阵
    np.save("./results/eval_phi.npy", phi)  # 保存phi矩阵
    return theta, phi


# hellinger距离计算
def hellinger_distance(t1, t2):
    return 1 / np.sqrt(2) * np.sqrt(np.sum(np.square(np.sqrt(t1) - np.sqrt(t2))))


# 评估准确率
def eval_model(train_theta, eval_theta, eval_set):
    print("开始评估模型")
    correct = 0
    for m in range(len(eval_theta)):  # 评估第m个验证数据
        distance = np.array([hellinger_distance(eval_theta[m], train_theta[i]) for i in range(train_M)])
        if np.argmin(distance) == eval_set[m][1]:  # 如果预测正确
            correct += 1
    print("准确率：", correct / eval_M)
    return


if __name__ == "__main__":
    corpus_path = "./chinese-corpus"  # 语料库路径
    tokens, labels = data_preprocess(corpus_path)  # 获取全部token和对应标签和token_id
    train_dataset, eval_dataset, eval_label = split_tokens(tokens, labels)  # 拆分全部token，抽样得到训练集和评估集
    token_len = np.zeros(train_M, dtype=np.int64)
    for a, b in train_dataset:
        token_len[b] += 1
    train_token_length = round(np.max(token_len) * train_len)  # 记录最长的token数
    train_dataset = concat_dataset(train_dataset)  # 将分离的数据集重新组合成嵌套list

    eval_dataset_no_concat = handle_eval_dataset(eval_dataset)  # 处理验证数据，但不将数据重新按文章组合

    token_map = generate_token_map(train_dataset, eval_dataset_no_concat)  # 根据训练数据生成token_map
    save_model_token_map(token_map)  # 保存token_map到文件
    V = len(token_map)

    t_nw, t_nwsum, t_nd, t_ndsum, t_theta, t_phi = train_model(train_dataset, 50 / K, 0.01, 20)  # 训练模型

    eval_token_length = eval_len  # 评估时不将各个段落整合，而是根据模型判断某个段落的采样属于哪篇文章
    e_theta, e_phi = predict_model(eval_dataset_no_concat, 50 / K, 0.01, 10, t_nw, t_nwsum)  # 在评估集上训练

    print("topic = {}, token = {}".format(K, sample_len))

    eval_model(t_theta, e_theta, eval_dataset)
