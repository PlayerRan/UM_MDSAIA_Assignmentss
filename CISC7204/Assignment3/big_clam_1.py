import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.special import expit  # Sigmoid function

# 创建一个Barabási-Albert图
G = nx.barabasi_albert_graph(30, 4)

# 绘制并显示图
nx.draw_networkx(G, with_labels=True, node_color='red', pos=nx.spring_layout(G, seed=0))
plt.savefig("./graph.jpg")
plt.show()

# BigCLAM模型实现
class BigClam:
    def __init__(self, G, num_communities, iterations=100, learning_rate=0.005):
        self.G = G
        self.num_communities = num_communities
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.F = np.random.rand(len(G.nodes), num_communities)  # 初始化社区成员矩阵

    def train(self):
        for _ in range(self.iterations):
            for u, v in self.G.edges():
                self.gradient_ascent(u, v)
            for u in self.G.nodes():
                self.gradient_ascent_negative(u)

    def gradient_ascent(self, u, v):
        common_membership = np.dot(self.F[u], self.F[v])
        prob = expit(common_membership)
        gradient = (1 - prob) * self.F[v]
        self.F[u] += self.learning_rate * gradient
        self.F[v] += self.learning_rate * gradient

    def gradient_ascent_negative(self, u):
        for v in self.G.nodes():
            if not self.G.has_edge(u, v):
                common_membership = np.dot(self.F[u], self.F[v])
                prob = expit(common_membership)
                gradient = -prob * self.F[v]
                self.F[u] += self.learning_rate * gradient

    def get_memberships(self):
        return self.F

# 训练Big Clam模型
num_communities = 2
bigclam = BigClam(G, num_communities)
bigclam.train()
memberships = bigclam.get_memberships()

# 使用KMeans对节点进行聚类
kmeans = KMeans(n_clusters=num_communities, random_state=0).fit(memberships)
labels = kmeans.labels_

# 打印每个节点的聚类标签
for i in range(len(G.nodes)):
    print(f'Node {i}: Community {labels[i]}')

# 根据聚类标签重新绘制图
nx.draw_networkx(G, with_labels=True, node_color=labels, pos=nx.spring_layout(G, seed=0))
plt.savefig("./graph_clustered.jpg")
plt.show()
