import tensorflow as tf
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
import matplotlib.pyplot as plt

# 合并（拼接）
a = tf.random.normal([4,35,8])
b = tf.random.normal([6,35,8])
c = tf.concat([a,b],axis = 0)


# 堆叠stack
a = tf.random.normal([35,8])
b = tf.random.normal([35,8])
d = tf.stack([a,b],axis=0)
# 合并与堆叠操作对于张量的维数有限制和要求

# 分割 num_or_size_splits 为整数时为等长分割，为list时可以不等长分割
a = tf.random.normal([10,35,8])
c = tf.split(a,axis = 0,num_or_size_splits= [4,2,2,2])
d = tf.unstack(a,axis=0)#特殊分割默认等分为1份的命令


# 5.2数据统计

#向量范数 一范数  二范数  无穷范数
a = tf.ones([2,2])
b = tf.norm(a,ord=2)


#最大值最小值、均值 tf.reduce_max tf.reduce_min tf.reduce_mean tf.reduce_sum
a = tf.random.normal([4,10])
b =tf.reduce_max(a,axis=1)
print(b[0])

















