import numpy as np

def solve_integral_problem():
    # 定义被积函数
    f = lambda x: np.sqrt(x + 1.5)
    exact_value = 2.399529

    print(f"被积函数: f(x) = sqrt(x + 1.5)")
    print(f"积分区间: [-1, 1]")
    print(f"积分准确值: {exact_value}\n")


    print("1. 牛顿-柯特斯求积公式 (Newton-Cotes)")

    # n=2: 梯形公式 (Trapezoidal Rule)
    # 节点: -1, 1; 权重: [1, 1] * (b-a)/2
    nc_n2 = (1 - (-1)) / 2 * (f(-1) + f(1))
    print(f"[n=2] 结果: {nc_n2:.6f}, 误差: {abs(nc_n2 - exact_value):.6e}")

    # n=3: 辛普森公式 (Simpson's Rule)
    # 节点: -1, 0, 1; 权重: [1, 4, 1] * (b-a)/6
    nc_n3 = (2 / 6) * (f(-1) + 4 * f(0) + f(1))
    print(f"[n=3] 结果: {nc_n3:.6f}, 误差: {abs(nc_n3 - exact_value):.6e}")

    # n=4: 辛普森 3/8 公式 (Simpson's 3/8 Rule)
    # 节点: -1, -1/3, 1/3, 1; 权重: [1, 3, 3, 1] * (b-a)*3/8 ... 注意这里分母处理
    # 标准形式: 3h/8 * (y0 + 3y1 + 3y2 + y3), h=(b-a)/3
    nodes_4 = np.linspace(-1, 1, 4)
    weights_4 = np.array([1, 3, 3, 1]) * (3 * ((1 - (-1))/3) / 8)
    nc_n4 = sum(w * f(x) for x, w in zip(nodes_4, weights_4))
    print(f"[n=4] 结果: {nc_n4:.6f}, 误差: {abs(nc_n4 - exact_value):.6e}")

    # n=5: 布尔公式 (Boole's Rule)
    # 节点: -1, -0.5, 0, 0.5, 1; 权重: [7, 32, 12, 32, 7] * (b-a)*2/45
    nodes_5 = np.linspace(-1, 1, 5)
    weights_5 = np.array([7, 32, 12, 32, 7]) * (2 * (1 - (-1)) / 45)
    nc_n5 = sum(w * f(x) for x, w in zip(nodes_5, weights_5))
    print(f"[n=5] 结果: {nc_n5:.6f}, 误差: {abs(nc_n5 - exact_value):.6e}")


    # ==========================================
    # 2. 高斯-勒让德求积公式 (Gauss-Legendre)
    # 使用 numpy 自动生成最优节点和权重
    # ==========================================
    print("\n" )
    print("2. 高斯-勒让德求积公式 (Gauss-Legendre)")


    # n=2
    nodes_2, weights_2 = np.polynomial.legendre.leggauss(2)
    gl_n2 = sum(w * f(x) for x, w in zip(nodes_2, weights_2))
    print(f"[n=2] 结果: {gl_n2:.6f}, 误差: {abs(gl_n2 - exact_value):.6e}")

    # n=3
    nodes_3, weights_3 = np.polynomial.legendre.leggauss(3)
    gl_n3 = sum(w * f(x) for x, w in zip(nodes_3, weights_3))
    print(f"[n=3] 结果: {gl_n3:.6f}, 误差: {abs(gl_n3 - exact_value):.6e}")

    # n=4
    nodes_4_g, weights_4_g = np.polynomial.legendre.leggauss(4)
    gl_n4 = sum(w * f(x) for x, w in zip(nodes_4_g, weights_4_g))
    print(f"[n=4] 结果: {gl_n4:.6f}, 误差: {abs(gl_n4 - exact_value):.6e}")

    # n=5
    nodes_5_g, weights_5_g = np.polynomial.legendre.leggauss(5)
    gl_n5 = sum(w * f(x) for x, w in zip(nodes_5_g, weights_5_g))
    print(f"[n=5] 结果: {gl_n5:.6f}, 误差: {abs(gl_n5 - exact_value):.6e}")


if __name__ == "__main__":
    solve_integral_problem()