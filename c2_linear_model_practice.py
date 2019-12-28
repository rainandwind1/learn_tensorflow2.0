import numpy as np
import matplotlib.pyplot as plt
# 采样数据
#y = 1.477*x + 0.089 + eps
loss_list = []
data = []
for i in range(100):
    x = np.random.uniform(-10.,10.)
    eps = np.random.normal(0.,0.1)
    y = 1.477*x + 0.089 + eps 
    data.append([x,y])

data = np.array(data)

# 计算均方差
def mse(b, w, points):
    totalError = 0
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (y - (w*x + b))**2

    return totalError/float(len(points))

# 计算梯度
def step_gradient(b_current, w_current, points, lr):
    b_gradient = 0
    w_gradient = 0
    M = float(len(points))
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += (2/M) * (w_current * x + b_current - y)
        w_gradient += (2/M) * ((w_current * x + b_current - y) * x)
         
    new_b = b_current - lr * b_gradient
    new_w = w_current - lr * w_gradient
    return [new_b,new_w]

# 梯度更新
def gradient_descent(points, starting_b, starting_w, lr, num_iterations):
    b = starting_b
    w = starting_w
    for step in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), lr)
        loss = mse(b, w, points)
        loss_list.append(loss)
        if step%50 == 0:
            print(f"iteration:{step}, loss:{loss}, w:{w}, b:{b}")
    return [b,w]


# 主函数
def main():
    lr = 0.01
    initial_b = 0
    initial_w = 0
    num_iterations = 1000
    [b,w] = gradient_descent(data, initial_b, initial_w, lr, num_iterations)
    loss = mse(b, w, data)
    print(f'Final loss:{loss}, w:{w}, b:{b}')

if __name__ == "__main__":
    main()    

plt.figure()
plt.plot([i for i in range(10)],loss_list[0:10])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

