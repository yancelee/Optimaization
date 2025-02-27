import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数
def f(x):
    x1, x2 = x
    return x1**2 + x2**2 + x1*x2 - x2

def gradient_f(x):
    x1, x2 = x
    grad_x1 = 2*x1 + x2
    grad_x2 = 2*x2 + x1 - 1
    return np.array([grad_x1, grad_x2])

# 梯度下降法
def gradient_descent():
    x = np.array([0, 0])
    learning_rate = 0.1
    path = [x]
    while True:
        grad = gradient_f(x)
        norm_grad = np.linalg.norm(grad, 1) # 用于计算向量或矩阵的范数
        if norm_grad < 1e-3:
            break
        x = x - learning_rate * grad
        path.append(x)
    return np.array(path)

path = gradient_descent()
optimal_x = path[-1]

# 生成绘制等高线的数据
x1_vals = np.linspace(-0.5, 0.8, 400)
x2_vals = np.linspace(-0.5, 0.8, 400) # 在指定的区间内生成等间距的一维数组
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X2.shape[1]):
        Z[i, j] = f([X1[i, j], X2[i, j]])

# 绘制等高线
plt.figure(figsize=(10, 8)) # plt.figure() 函数用于创建一个新的图形窗口
# plt.contour() 函数会自动根据 Z 数组中的值，将函数值的范围分成 50 个等级，并绘制相应的等高线
plt.contour(X1, X2, Z, 50, cmap='viridis') # 'viridis' 是 matplotlib 提供的一种渐变色，方便用户直观地看出函数值的大小分布
plt.colorbar() # plt.colorbar() 函数用于在图形中添加一个颜色条，通过颜色条可以直观地了解等高线所代表的函数值的大小

# 绘制梯度下降路径
# r：代表线条的颜色为红色
# 实线 -，还可以使用其他线条样式，例如虚线 --、点划线 -. 等
# 除了圆形 o，还有方形 s、三角形 ^ 等多种标记样式可供选择
plt.plot(path[:, 0], path[:, -1], 'r-o', label='Gradient Descent Path') # path的每一行代表一次迭代， 第一列是x1，第二列是x2

# 标记初始点和最优解
plt.scatter(path[0, 0], path[0, 1], color='blue', marker='s', label='Initial Point')
# s 是 plt.scatter() 函数的一个参数，用于指定标记点的大小。
plt.scatter(optimal_x[0], optimal_x[1], color='green', marker='*', s=20, label='Optimal Solution')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Gradient Descent for $f(x_1, x_2) = x_1^2 + x_2^2 + x_1x_2 - x_2$')
plt.legend()
plt.grid(True)
plt.show()

print('最优解 x:', optimal_x)
print('最优值 f(x):', f(optimal_x))