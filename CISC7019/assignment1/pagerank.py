import networkx as nx
import numpy as np

def read_graph_from_input():
    # 从输入读取图的边
    edges = []
    print("请输入图的边（格式：节点1 节点2），输入 'done' 结束输入：")
    while True:
        edge = input()
        if edge.lower() == 'done':
            break
        nodes = edge.split()
        if len(nodes) == 2:
            edges.append((nodes[0], nodes[1]))
        else:
            print("输入格式错误，请重新输入。")

    return edges

def build_adjacency_matrix(edges, nodes):
    #创建邻接矩阵
    N = len(nodes)
    M = np.zeros((N, N))
    node_index = {node: i for i, node in enumerate(nodes)}

    for edge in edges:
        i, j = node_index[edge[0]], node_index[edge[1]]
        M[j, i] = 1

    for i in range(N):
        if M[:, i].sum() != 0:
            M[:, i] /= M[:, i].sum()

    return M, node_index

def compute_pagerank(edges, beta=0.8, max_iter=100, tol=1.0e-6):
    #计算PageRank值
    nodes = list(set([node for edge in edges for node in edge]))
    N = len(nodes)
    M, node_index = build_adjacency_matrix(edges, nodes)
    PR = np.ones(N) / N

    for _ in range(max_iter):
        new_PR = (1 - beta) / N + beta * M.dot(PR)
        if np.linalg.norm(new_PR - PR) < tol:
            break
        PR = new_PR

    pagerank = {node: PR[node_index[node]] for node in nodes}
    return pagerank


def main():
    # 读取图的边
    edges = read_graph_from_input()

    # 计算PageRank值
    pagerank = compute_pagerank(edges)

    # 输出每个页面的PageRank值
    print("\nPageRank of each page:")
    for page, rank in pagerank.items():
        print(f"{page}: {rank}")

if __name__ == "__main__":
    main()
