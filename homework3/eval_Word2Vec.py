import gensim.models
import gensim.corpora
import sklearn.cluster
import numpy as np
import sklearn.manifold
import matplotlib.pyplot as plt

import utility

def main():
    model = gensim.models.Word2Vec.load("./result/Word2Vec/model.bin")

    # 任务1，词向量间语义距离，选取相似度最高的词输出，判断分类效果
    word_list = ["轻功", "魔教", "群豪", "无礼", "小弟"]
    for word in word_list:
        similar_words = model.wv.most_similar(word)
        print("与{}最相似的词有：".format(word))
        for v in similar_words:
            print(v)

    # 任务2，词向量聚类
    words = list(model.wv.key_to_index) # 选择词
    words = words[0:500] # 选取前500个词
    word_vectors = np.array([model.wv[word] for word in words]) # 提取词向量

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
    paragraph_vector1 = utility.get_paragraph_vector_Word2Vec(model, paragraph1)
    paragraph_vector2 = utility.get_paragraph_vector_Word2Vec(model, paragraph2)
    paragraph_vector3 = utility.get_paragraph_vector_Word2Vec(model, paragraph3)
    # 获取文章12,13,23之间的余弦相似度
    s12 = utility.cosine_distance(paragraph_vector1, paragraph_vector2)
    s13 = utility.cosine_distance(paragraph_vector1, paragraph_vector3)
    s23 = utility.cosine_distance(paragraph_vector2, paragraph_vector3)
    print("文章1和文章2的余弦相似度为{}".format(s12))
    print("文章1和文章3的余弦相似度为{}".format(s13))
    print("文章2和文章3的余弦相似度为{}".format(s23))

if __name__ == "__main__":
    main()