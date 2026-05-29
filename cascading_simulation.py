import random


def generate_random_network(num_nodes, edge_prob=0.2):
    """
    随机生成一个无向网络（邻接表格式）
    :param num_nodes: int, 节点总数
    :param edge_prob: float, 两个节点之间存在连边的概率 (0~1)
    """
    network = {i: [] for i in range(num_nodes)}

    # 遍历所有可能的节点对，以 edge_prob 的概率决定是否连边
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                network[i].append(j)
                network[j].append(i)

    # 【防御性处理】：防止出现完全孤立的节点（度为0会导致无法扩散）
    for node in network:
        if len(network[node]) == 0 and num_nodes > 1:
            # 如果该节点是孤岛，随机给它找一个邻居连上
            neighbor = random.choice([n for n in network if n != node])
            network[node].append(neighbor)
            network[neighbor].append(node)

    return network


def simulate_cascade(network, initial_nodes, threshold):
    """
    模拟线性阈值模型下的网络级联扩散过程
    """
    # 初始化状态字典：True表示已接受，False表示未接受
    status = {node: False for node in network}
    time_to_accept = {}  # 记录各个节点从知晓到行动的延时

    # 1. 设置初始激活节点集合
    activated_set = set()
    for node in initial_nodes:
        if node in network:
            status[node] = True
            time_to_accept[node] = 0  # 初始节点延时记为0
            activated_set.add(node)

    current_step = 1
    print(f"=== Step {current_step - 1} ===")
    print(f"Initial Activated Nodes: {sorted(list(activated_set))}")

    # 2. 循环模拟扩散过程
    while True:
        newly_activated = []

        # 遍历所有未激活的节点
        for node in network:
            if not status[node]:
                neighbors = network[node]
                if len(neighbors) == 0: continue

                # 计算已激活邻居占总邻居的比例
                activated_neighbors_count = sum(1 for n in neighbors if status[n])
                ratio = activated_neighbors_count / len(neighbors)

                # 【关键】使用 >= 判断是否达到门槛值
                if ratio >= threshold:
                    status[node] = True
                    time_to_accept[node] = current_step
                    newly_activated.append(node)

        # 如果本轮没有新节点被激活，说明扩散达到稳态，退出循环
        if not newly_activated:
            break

        # 更新全局激活集合并输出当前步骤的动态结果
        activated_set.update(newly_activated)
        print(f"\n=== Step {current_step} ===")
        print(f"Newly Activated: {sorted(newly_activated)}")
        print(f"Total Activated So Far: {len(activated_set)} / {len(network)}")

        current_step += 1

    # 3. 计算平均延时
    total_time = sum(time_to_accept.values())
    avg_delay = total_time / len(time_to_accept) if time_to_accept else 0

    return time_to_accept, avg_delay


# ================= 主程序运行 =================
if __name__ == "__main__":
    # 参数配置区
    N = 15  # 随机生成的节点总数
    P = 0.15  # 连边概率 (稀疏网络设为0.1~0.2左右比较合理)
    THRESHOLD = 0.3  # 采纳门槛值 (0~1之间，越低越容易扩散)
    NUM_INIT = 2  # 随机选取的初始种子节点数量

    # 1. 随机生成网络
    net = generate_random_network(N, P)
    print("Generated Network Adjacency List:")
    for k, v in net.items():
        print(f"Node {k}: Neighbors {v}")
    print("-" * 40)

    # 2. 随机选择初始节点
    init_nodes = random.sample(list(net.keys()), min(NUM_INIT, N))

    # 3. 执行级联扩散模拟
    delays, average_delay = simulate_cascade(net, init_nodes, THRESHOLD)

    # 4. 输出最终统计结果
    print("\n--- Simulation Finished ---")
    print("Time to accept for each node:")
    for node, t in sorted(delays.items()):
        print(f"Node {node}: Delay = {t}")
    print(f"\nAverage Delay from Awareness to Action: {average_delay:.2f}")