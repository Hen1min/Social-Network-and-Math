import matplotlib.pyplot as plt
import numpy as np

def generate_adjacency_matrix(n):
    if n < 0:
        return None

    adjacency_matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < 0.5:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    return adjacency_matrix

def degree_of_node_bool(adj_matrix, node_index):
    if node_index < 0 or node_index >= adj_matrix.shape[0]:
        return None

    neighbors = np.where(adj_matrix[node_index] == 1)[0]

    if len(neighbors) == 0:
        return 0

    sum = 0
    for i in neighbors:
        sum += np.sum(adj_matrix[i])

    avg = sum / len(neighbors)
    if len(neighbors) > avg:
        return 0
    else:
        return 1


# 主函数
def main():
    num_ratio_average = []
    num_list = []
    for i in range(100,1000,100):
        num_list.append(i)

    num_list.extend([1000])

    for num in num_list:
        print(f"num : {num}")

        list_ratio = []

        for i in range(0, 5):
            this_matrix = generate_adjacency_matrix(num)
            fit = 0
            for index in range(0, num):
                fit += degree_of_node_bool(this_matrix, index)
            list_ratio.append(fit / num)

        average = sum(list_ratio) / len(list_ratio)
        num_ratio_average.append(average)

    print(num_ratio_average)
    print(num_list)

    plt.plot(num_list, num_ratio_average, marker='o', linestyle='-')
    plt.title('Certification of Friendship Paradox')
    plt.xlabel('Number of People in the Network')
    plt.ylabel('Ratio of Paradox')

    plt.show()

if __name__ == "__main__":
    main()