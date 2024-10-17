import networkx as nx
import numpy as np
from scipy.special import expit  # Sigmoid function

# 创建示例图（假设Example 10.22包含4个节点和4条边）
G = nx.Graph()
edges = [('w', 'x'), ('x', 'y'), ('y', 'z'), ('w', 'y')]
G.add_edges_from(edges)

# 定义社区成员关系
community_guesses = [
    {'C1': {'w', 'x'}, 'C2': {'y', 'z'}},
    {'C1': {'w', 'x', 'y', 'z'}, 'C2': {'x', 'y', 'z'}}
]

def calculate_edge_probabilities(communities, pC1, pC2):
    probabilities = {}
    for u, v in G.edges():
        if u in communities['C1'] and v in communities['C1']:
            prob = pC1
        elif u in communities['C2'] and v in communities['C2']:
            prob = pC2
        elif (u in communities['C1'] and v in communities['C2']) or (u in communities['C2'] and v in communities['C1']):
            prob = pC1 + pC2 - pC1 * pC2
        else:
            prob = 0  # ǫ, 表示非常小的概率
        probabilities[(u, v)] = prob
    return probabilities

def calculate_mle(communities):
    # 假设的pC1和pC2值
    pC1 = 0.5
    pC2 = 0.3

    # 计算每条边的连接概率
    edge_probs = calculate_edge_probabilities(communities, pC1, pC2)
    
    # 计算MLE
    log_likelihood = 0
    for (u, v), prob in edge_probs.items():
        if G.has_edge(u, v):
            log_likelihood += np.log(prob)
        else:
            log_likelihood += np.log(1 - prob)
    
    return log_likelihood

def main():
    for i, communities in enumerate(community_guesses):
        mle = calculate_mle(communities)
        print(f"MLE for guess {i+1}: {mle}")
#   (cisc7201) D:\Mycode\UM_MDSAIA\UM_MDSAIA_Assignmentss\CISC7204\Assignment3>python big_clam_2.py
#   MLE for guess 1: -2.7586858170707895
#   MLE for guess 2: -2.772588722239781
if __name__ == "__main__":
    main()
