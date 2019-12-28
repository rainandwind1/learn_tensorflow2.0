import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
import matplotlib.pyplot as plt
tf.enable_eager_execution()
# # 4.1 数据类型
# a = 1.2
# aa = tf.constant(1.2,dtype=tf.float32)
# aa = tf.cast(aa,tf.float64)#类型转换  类型转换时非零数据否视为true 
# ae = tf.constant('Hello world')
# c = tf.constant(True)     
# #print(aa.dtype)
# #待优化张量 tf.Variable  不需要计算梯度的变量不要用
# # tf.Variable封装，可以节省计算资源
# a = tf.constant([-1, 1, 0, 2])
# aa = tf.Variable(a) # 标志这个变量是可以训练的，需要被优化
# v = np.matrix([[1,2,3]])
# #print(v,aa.name,aa.trainable)
# a = tf.convert_to_tensor(np.array([[1,2.],[1,2]]))
# # tf.constant 和 tf.convert_to_tensor都是新建tensor的命令

# a = tf.ones(shape=[2,2])
# c = tf.ones_like(a)
# c = tf.fill([3,3],3)
# b = tf.random.normal(shape=[2,3],mean=3,stddev=1)
# b = tf.random.uniform(shape=[2,5],maxval=10,dtype=tf.int32)
# c = tf.range(10,delta=1)
# # Accuracy  Precision Recall
# print(c)
lr = 0.1
epoch = 100
loss_list = []
(x,y),(x_test,y_test) = datasets.mnist.load_data()
x = 2.*tf.convert_to_tensor(x,dtype=tf.float32)/255. - 1.0
y = tf.convert_to_tensor(y,dtype=tf.int32)
y_onehot = tf.one_hot(y,depth=10)

for i in range(epoch):
    with tf.GradientTape() as tape:
        w1 = tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1))
        b1 = tf.Variable(tf.zeros([256]))

        w2 = tf.Variable(tf.random.truncated_normal([256,128],stddev=0.1))
        b2 = tf.Variable(tf.zeros([128]))

        w3 = tf.Variable(tf.random.truncated_normal([128,10],stddev=0.1))
        b3 = tf.Variable(tf.zeros([10]))

        x = tf.reshape(x,[-1,28*28])

        y1 = tf.nn.relu(x@w1 + b1) 
        y2 = tf.nn.relu(y1@w2 + b2)
        out = y2@w3 + b3

        loss = tf.square(y_onehot - out)
        loss = tf.reduce_mean(loss)
        loss_list.append(loss)

    grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
    w1.assign_sub(lr * grads[0])
    b1.assign_sub(lr * grads[1])
    w2.assign_sub(lr * grads[2])
    b2.assign_sub(lr * grads[3])
    w3.assign_sub(lr * grads[4])
    b3.assign_sub(lr * grads[5])


plt.figure()
plt.plot([i for i in range(epoch)],loss_list)
plt.show()



