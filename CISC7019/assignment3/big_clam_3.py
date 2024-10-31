import networkx as nx
import numpy as np
from scipy.special import expit  # Sigmoid function

# 创建示例图（假设Example 10.22包含4个节点和4条边）
G = nx.Graph()
edges = [('w', 'x'), ('x', 'y'), ('y', 'z'), ('w', 'y')]
G.add_edges_from(edges)

# 初始社区成员关系
communities = {'C': {'w', 'x', 'y'}, 'D': {'w', 'y', 'z'}}

def calculate_edge_probabilities(communities, pC, pD):
    probabilities = {}
    for u, v in G.edges():
        if u in communities['C'] and v in communities['C']:
            prob = pC
        elif u in communities['D'] and v in communities['D']:
            prob = pD
        elif (u in communities['C'] and v in communities['D']) or (u in communities['D'] and v in communities['C']):
            prob = pC + pD - pC * pD
        else:
            prob = 1e-10  # ǫ, 表示非常小的概率
        probabilities[(u, v)] = prob
    return probabilities

def calculate_mle(communities):
    # 假设的pC和pD值
    pC = 0.5
    pD = 0.3

    # 计算每条边的连接概率
    edge_probs = calculate_edge_probabilities(communities, pC, pD)
    
    # 计算MLE
    log_likelihood = 0
    for (u, v), prob in edge_probs.items():
        if G.has_edge(u, v):
            log_likelihood += np.log(prob)
        else:
            log_likelihood += np.log(1 - prob)
    
    return log_likelihood

def optimize_communities(communities):
    nodes = set(G.nodes())
    best_mle = calculate_mle(communities)
    best_communities = communities.copy()
    
    improved = True
    while improved:
        improved = False
        for node in nodes:
            for community in ['C', 'D']:
                new_communities = {k: v.copy() for k, v in communities.items()}
                if node in new_communities[community]:
                    new_communities[community].remove(node)
                else:
                    new_communities[community].add(node)
                
                new_mle = calculate_mle(new_communities)
                if new_mle > best_mle:
                    best_mle = new_mle
                    best_communities = new_communities
                    improved = True
                    break
            if improved:
                break
        communities = best_communities
    
    return best_communities, best_mle

def main():
    final_communities, final_mle = optimize_communities(communities)
    print(f"Final communities: {final_communities}")
    print(f"Final MLE: {final_mle}")

if __name__ == "__main__":
    main()

# (cisc7201) D:\Mycode\UM_MDSAIA\UM_MDSAIA_Assignmentss\CISC7204\Assignment3>python big_clam_3.py
# Final communities: {'C': {'y', 'x'}, 'D': {'w', 'z'}}
# Final MLE: -1.9854959288373077
