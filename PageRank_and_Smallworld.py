import numpy as np


# 1. 生成随机有向矩阵
def generate_matrix(n, p=0.02): # 默认连边概率改为 2%
    matrix = np.random.random((n, n)) < p # 使用概率 p 生成
    matrix = matrix.astype(int)
    np.fill_diagonal(matrix, 0)
    return matrix


# 2. 转为无向矩阵
def to_undirected(matrix):
    return np.clip(matrix + matrix.T, 0, 1)


# 3. 计算 PageRank 向量
def calculate_pagerank(matrix, d=0.85):
    n = matrix.shape[0]
    out_deg = matrix.sum(axis=1)
    out_deg[out_deg == 0] = 1
    M = (matrix.T / out_deg).astype(float)
    M[:, (matrix.sum(axis=1) == 0)] = 1.0 / n
    M = d * M + (1 - d) / n

    pr = np.ones(n) / n
    for _ in range(100):
        pr = M.dot(pr)
    return pr


# 4. 计算六度空间占比
def calculate_six_degrees(undirected_matrix):
    n = undirected_matrix.shape[0]
    percentages = []
    for start in range(n):
        visited, queue = [False] * n, [(start, 0)]
        visited[start], count = True, 1
        while queue:
            node, dist = queue.pop(0)
            if dist < 6:
                for neighbor in range(n):
                    if undirected_matrix[node][neighbor] == 1 and not visited[neighbor]:
                        visited[neighbor], count = True, count + 1
                        queue.append((neighbor, dist + 1))
        percentages.append(count / n * 100)
    return np.mean(percentages)  # 返回所有节点的平均占比


# --- 主程序 ---
if __name__ == "__main__":
    # 1. 输出 10个节点时的有向矩阵
    n_10_matrix = generate_matrix(10)
    print("【10个节点时的有向矩阵】：")
    print(n_10_matrix)
    print("\n" + "-" * 40 + "\n")

    # 2. 输出 10个节点时的 PageRank 向量
    pr_vector = calculate_pagerank(n_10_matrix)
    print("【10个节点时的 PageRank 向量】：")
    print(np.round(pr_vector, 6))
    print("\n" + "-" * 40 + "\n")

    # 3. 输出不同节点数下的六度空间占比表格
    node_sizes = [10, 100, 500, 1000]
    print("【不同节点数下的六度空间平均占比】：")
    print(f"{'节点数':<10} | {'平均占比 (%)':<15}")
    print("-" * 30)

    for n in node_sizes:
        # 每次重新生成对应规模的矩阵
        matrix = generate_matrix(n)
        undirected_matrix = to_undirected(matrix)
        avg_percentage = calculate_six_degrees(undirected_matrix)
        print(f"{n:<10} | {avg_percentage:<15.2f}")