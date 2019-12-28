import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from sklearn.datasets import make_moons 
#import seaborn as sns
from sklearn.model_selection import train_test_split
tf.enable_eager_execution()



# 训练集数据采集和处理
(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
X_train = []
Y_train = []
X_test = []                     
Y_test = []
for k,tag in enumerate(y_train):
    if tag == 1:
        X_train.append(x_train[k])
        Y_train.append(y_train[k])
for k,tag in enumerate(y_train):
    if tag == 9:
        X_train.append(x_train[k])
        Y_train.append(y_train[k])
X_train = tf.constant(X_train)
Y_train = tf.constant(Y_train)

print(X_train.shape,Y_train.shape)

# 测试集采集数据和处理
for k,tag in enumerate(y_test):
    if tag == 1:
        X_test.append(x_test[k])
        Y_test.append(y_test[k])
for k,tag in enumerate(y_test):
    if tag == 9:
        X_test.append(x_test[k])
        Y_test.append(y_test[k])
X_test = tf.constant(X_test)
Y_test = tf.constant(Y_test)
print(X_test.shape,Y_test[3].numpy())
# def himmelblau(x):
#     #himmelblau函数实现
#     return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


# # x = np.arange(-6.,6,0.1)
# # y = np.arange(-6,6,0.1)
# # X,Y = np.meshgrid(x,y)
# # #print(X[0][0],Y[0][0])
# # Z = himmelblau([X,Y])
# # fig = plt.figure('himmelblau')
# # ax = fig.gca(projection='3d')
# # ax.plot_surface(X,Y,Z)
# # ax.view_init(60,-30)
# # ax.set_xlabel('x')
# # ax.set_ylabel('y')
# # plt.show()



# #梯度下降法求解himmelblua函数最小值点

# # x = tf.constant([4.0,0.0])
# # for step in range(200):
# #     with tf.GradientTape() as tape:
# #         tape.watch([x])
# #         y = himmelblau(x)
# #         grads = tape.gradient(y,[x])[0]
# #         x -= 0.01*grads
# #         if step%20 == 19:
# #             print('step{}:x={},f(x) = {}'.format(step,x.numpy(),y.numpy()))
        
# # 反向传播算法实战
# N_SAMPLES = 2000
# TEST_SIZE = 0.3
# x, y =  make_moons(n_samples = N_SAMPLES,noise = 0.2,random_state = 100)
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = TEST_SIZE,random_state = 42)

# # def make_plot(x,y,plot_name,file_name = None,XX=None,YY=None,preds = None,dark = False):
# #     if(dark):
# #         plt.style.use('dark_background')
# #     # else:
# #     #     sns.set_style('whitegrid')
# #     plt.figure(figsize=(16,12))
# #     axes = plt.gca()
# #     axes.set(xlabel = "$x_1$",ylabel="$x_2$")
# #     plt.title(plot_name,fontsize = 30)
# #     plt.subplots_adjust(left = 0.2)
# #     plt.subplots_adjust(right = 0.8)
# #     if(XX is not None and YY is not None and preds is not None):
# #         plt.contourf(XX,YY,preds.reshape(XX.shape),25,aplha = 1,cmap = plt.cm.Spectral)
# #         plt.contour(XX,YY,preds.reshape(XX.shape),levels = [.5],cmap = 'Greys',xmin = 0,vmax = .6)
# #     plt.scatter(x[:,0],x[:,1],c = y.ravel(),s = 40,cmap = plt.cm.Spectral,edgecolors = 'None')
# #     plt.savefig('dataset.svg')
# #     #plt.close()

# # make_plot(x,y,"Classification Dataset Visualization")
# # plt.show()


# #新建网络层
# class Layer:
#     def __init__(self,n_input,n_neurons,actiovation = None,weights=None,bias = None):
#         self.weights = weights if weights is not None else np.random.randn(n_input,n_neurons)*np.sqrt(1/n_neurons)
#         self.bias = bias if bias is not None else np.random.rand(n_neurons)*0.1
#         self.activation  = actiovation
#         self.last_activation = None
#         self.error = None
#         self.delta = None
    
#     def activate(self,x):
#         r = np.dot(x,self.weights) + self.bias
#         self.last_activation = self._apply_activation(r)
#         return self.last_activation

#     def _apply_activation(self,r):
#         if self.activation is None:
#             return r
#         elif self.activation == 'relu':
#             return tf.nn.relu(r)
#         elif self.activation == 'tanh':
#             return np.tanh(r)
#         elif self.activation == 'sigmoid':
#             return tf.nn.sigmoid(r)
#         return r

#     def apply_activation_derivative(self,r):
#         # 计算激活函数的导数
#         # 无激活函数，导数为1
#         if self.activation is None:
#             return np.ones_like(r)
#         # ReLU 函数的导数实现
#         elif self.activation == 'relu':
#             grad = np.array(r, copy=True)
#             grad[r > 0] = 1.
#             grad[r <= 0] = 0.
#             return grad
#         # tanh 函数的导数实现
#         elif self.activation == 'tanh':
#             return 1 - r ** 2
#         # Sigmoid 函数的导数实现
#         elif self.activation == 'sigmoid':
#             return r * (1 - r)
#         return r

# # 网络模型
# class NeuralNetwork:
#     def __init__(self):
#         self._layers = []

#     def add_layers(self,layer):
#         self._layers.append(layer)

#     def feed_forward(self,x):
#         for layer in self._layers:
#             x = layer.activate(x)
#         return x


#     #反向传播误差更新权值
#     def backpropagation(self,x,y,learning_rate):
#             output  = self.feed_forward(x) #计算输出
#             for i in reversed(range(len(self._layers))):
#                 layer = self._layers[i]
#                 if layer == self._layers[-1]:
#                     layer.error = y - output
#                     layer.delta = layer.error*layer.apply_activation_derivative(output)
#                 else:
#                     next_layer = self._layers[i + 1]
#                     # 下一层的误差加权和
#                     layer.error = np.dot(next_layer.weights,next_layer.delta)
#                     # 套上这层输出的导数
#                     layer.delta = layer.error*layer.apply_activation_derivative(layer.last_activation)
#                     # 乘上输出点
#                     for i in range(len(self._layers)):
#                         layer = self._layers[i]
#                         o_i = np.atleast_2d(x if i == 0. else self._layers[i - 1].last_activation)
#                         layer.weights += layer.delta*o_i.T*learning_rate

#     def accuracy(self,out_data,in_data):
#         return 0
                    
#     def train(self,x_train,x_test,y_train,y_test,learning_rate,max_epochs):
#         # onehot 编码
#         y_onehot = np.zeros((y_train.shape[0],2))
#         y_onehot[np.arange(y_train.shape[0]),y_train] = 1
#         mses = []
#         for i in range(max_epochs):
#             for j in range(len(x_train)):
#                 self.backpropagation(x_train[j],y_onehot[j],learning_rate)
#             if i%10 == 0:
#                 mse = np.mean(np.square(y_onehot - self.feed_forward(x_train)))
#                 mses.append(mse)
#                 print('Epoch:#%s, MSE:%f' % (i,float(mse)))

#                 #print('Accurecy:%.2f%%' % (self.accuracy(self.predict(x_test),y_test.flatten())*100))
#         return mses



# # #建立神经网络
# nn = NeuralNetwork()
# nn.add_layers(Layer(2,25,'sigmoid'))
# nn.add_layers(Layer(25,50,'sigmoid'))
# nn.add_layers(Layer(50,25,'sigmoid'))
# nn.add_layers(Layer(25,2,'sigmoid'))
# nn.train(x_train,x_test,y_train,y_test,learning_rate = 0.1,max_epochs = 200)


















