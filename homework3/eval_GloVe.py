import numpy as np
import gensim.corpora
import sklearn.cluster
import sklearn.manifold
import matplotlib.pyplot as plt
import collections

import utility

def main():
    embeddings = np.load('./result/GloVe/embeddings.npy') # 读取embeddings矩阵
    dictionary_high_freq = gensim.corpora.Dictionary.load('./result/GloVe/filtered_dictionary.bin') # 读取词典
    
    # 任务1，词向量间语义距离，选取相似度最高的词输出，判断分类效果
    word_list = ["轻功", "魔教", "群豪", "无礼", "小弟"]
    for word in word_list:
        similar_words = utility.most_similar(dictionary_high_freq, embeddings, word, 10)
        print("与{}最相似的词有：".format(word))
        for v in similar_words:
            print(v[1])

    # 任务2，词向量聚类
    word_freq = collections.Counter(dictionary_high_freq.dfs)
    # 按照词频从高到低排序，并获取前500个高频词
    top_words = word_freq.most_common(500)

    # 仅保留词本身，忽略词频，放入一个list中
    word_ids = [word for word, freq in top_words]
    words = [dictionary_high_freq[id] for id in word_ids]
    word_vectors = np.array([embeddings[id] for id in word_ids]) # 提取词向量

    # 应用K-means聚类
    num_clusters = 10  # 假设分为10类
    kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters)
    kmeans.fit(word_vectors)

    labels = kmeans.labels_ # 为每个词分配聚类标签

    for i in range(num_clusters):
        for j in range(500):
            if labels[j] == i:
                print("单词：{}，类别：{}".format(words[j], i))

    # 使用t-SNE降维
    tsne_model = sklearn.manifold.TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000, random_state=42)
    new_values = tsne_model.fit_transform(word_vectors)

    # 绘制聚类结果
    x = new_values[:, 0]
    y = new_values[:, 1]
    plt.figure(figsize=(10, 10))
    for i in range(num_clusters):
        plt.scatter(x[labels == i], y[labels == i], label=f"Cluster {i}")
    plt.legend()
    plt.show()

    # 任务3，段落语义关联
    # 获取段落
    paragraph1 = utility.get_paragraph("./paragraph/1.txt")
    paragraph2 = utility.get_paragraph("./paragraph/2.txt")
    paragraph3 = utility.get_paragraph("./paragraph/3.txt")
    # 获取段落向量
    paragraph_vector1 = utility.get_paragraph_vector_GloVe(dictionary_high_freq,embeddings, paragraph1)
    paragraph_vector2 = utility.get_paragraph_vector_GloVe(dictionary_high_freq,embeddings, paragraph2)
    paragraph_vector3 = utility.get_paragraph_vector_GloVe(dictionary_high_freq,embeddings, paragraph3)
    # 获取文章12,13,23之间的余弦相似度
    s12 = utility.cosine_distance(paragraph_vector1, paragraph_vector2)
    s13 = utility.cosine_distance(paragraph_vector1, paragraph_vector3)
    s23 = utility.cosine_distance(paragraph_vector2, paragraph_vector3)
    print("文章1和文章2的余弦相似度为{}".format(s12))
    print("文章1和文章3的余弦相似度为{}".format(s13))
    print("文章2和文章3的余弦相似度为{}".format(s23))
    

if __name__ == "__main__":
    main()