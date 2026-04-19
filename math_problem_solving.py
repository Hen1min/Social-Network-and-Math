import numpy as np
import matplotlib.pyplot as plt

# --- 1. 准备数据 ---
x = np.array([0.5, 1.1, 1.7, 2.1, 2.5, 2.9, 3.3, 3.7, 4.2, 4.9, 5.3, 6.0])
y = np.array([1.6, 2.4, 3.8, 4.3, 4.7, 4.8, 5.5, 6.1, 6.3, 7.1, 7.4, 8.2])

# --- 2. 数据中心化 ---
x_mean = np.mean(x)
y_mean = np.mean(y)

x_centered = x - x_mean
y_centered = y - y_mean

# --- 3. 构建增广矩阵 M ---
M = np.column_stack((x_centered, y_centered))

# --- 4. SVD 分解 ---
U, S, Vt = np.linalg.svd(M)

# --- 5. 提取解向量 ---
v_min = Vt[-1, :]  # 这是一个包含 2 个元素的向量 [a, b]

print(f"SVD 得到的法向量 (a, b): {v_min}")

# --- 6. 计算直线参数 ---
k_tls = -v_min[0] / v_min[1]
b_tls = y_mean - k_tls * x_mean

print(f"完全最小二乘法拟合结果: y = {k_tls:.4f}x + {b_tls:.4f}")

# --- 7. 计算残差的 2-范数 (垂直距离) ---
A_line = k_tls
B_line = -1
C_line = b_tls

# 计算所有点的距离
distances = np.abs(A_line * x + B_line * y + C_line) / np.sqrt(A_line**2 + B_line**2)
norm_2 = np.linalg.norm(distances)

print(f"残差的 2-范数 (正交距离和): {norm_2:.4f}")

# --- 8. 绘图验证 ---
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Data Points')

# 绘制 TLS 拟合直线
x_line = np.linspace(0, 7, 100)
y_line = k_tls * x_line + b_tls
plt.plot(x_line, y_line, color='green', linewidth=2, label=f'TLS Fit (SVD)')

# 绘制普通最小二乘 (OLS) 做对比
A_ols = np.column_stack((x, np.ones(len(x))))
sol_ols = np.linalg.lstsq(A_ols, y, rcond=None)[0]
y_ols = sol_ols[0] * x_line + sol_ols[1]
plt.plot(x_line, y_ols, color='blue', linestyle='--', label=f'OLS Fit (Vertical)')

plt.legend()
plt.grid(True)
plt.title(f'TLS via SVD: y={k_tls:.2f}x+{b_tls:.2f}')
plt.show()