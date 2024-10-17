import networkx as nx
from networkx.algorithms.community import girvan_newman
import matplotlib.pyplot as plt

# 创建示例图（Karate Club图）
G = nx.karate_club_graph()

# 使用Girvan-Newman算法将图划分为K个社区的函数
def girvan_newman_partition(G, K):
    comp = girvan_newman(G)
    limited = next(comp)
    while len(limited) < K:
        limited = next(comp)
    return limited

# 将图划分为K个社区
K = 3
communities = girvan_newman_partition(G, K)

# 绘制带有社区的图
pos = nx.spring_layout(G)
colors = ['r', 'g', 'b', 'y', 'c', 'm']
for i, community in enumerate(communities):
    nx.draw_networkx_nodes(G, pos, nodelist=list(community), node_color=colors[i % len(colors)])
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)
plt.show()

# 打印社区
for i, community in enumerate(communities):
    print(f"Community {i+1}: {list(community)}")
