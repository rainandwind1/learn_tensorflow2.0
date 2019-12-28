import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from sklearn.datasets import make_moons
from tensorflow import keras
from tensorflow.keras import layers 
#import seaborn as sns
from sklearn.model_selection import train_test_split
tf.enable_eager_execution()

# x = tf.constant([2.,1.,0.1])
# layer = layers.Softmax(axis = -1)
# print(layer(x).numpy())


# from tensorflow.keras import layers,Sequential,Model

# # network =  Sequential([
# #     layers.Dense(3,activation = None),
# #     layers.ReLU(),
# #     layers.Dense(2,activation = None),
# #     layers.ReLU()
# # ])

# x = tf.random.normal([4,3])
# #print(network(x))
# # layers_num = 2
# # network = Sequential([])
# # for _ in range(layers_num):
# #     network.add(layers.Dense(3))
# #     network.add(layers.ReLU())
# # network.build(input_shape = (None,4))
# # network.summary()

# # for p in network.trainable_variables:
# #     print(p.name,p.shape)

# #模型装配
# network = Sequential([
#     layers.Dense(256,activation = 'relu'),
#     layers.Dense(128,activation = 'relu'),
#     layers.Dense(64,activation = 'relu'),
#     layers.Dense(32,activation = 'relu'),
#     layers.Dense(10,activation = None)
# ])

from tensorflow.keras import optimizers,losses,datasets
# network.compile(optimizer = tf.train.AdamOptimizer(0.01),loss = losses.categorical_crossentropy,metrics = ['accuracy'])

# #模型训练
# x = tf.random.normal([4,4])
# y = tf.random.normal([4])
# x_test = tf.random.normal([4,4])
# y_test =tf.random.normal([4])
# history = network.fit(x,y,epochs = 20,validation_data = (x_test,y_test),validation_steps = 1)

# # 模型测试
# x, y = next(iter(db_test))
# print('predict x:',x.shape)
# out = network.predict(x)
# print(out)

# 模型保存与加载 export_saved_model
# tf.keras.models.save_model(network, 'model-savedmodel')
# print('export saved model.')

# resnet = keras.applications.ResNet50(weights='imagenet',include_top=False)
# resnet.summary()
from tensorflow.keras import metrics
# loss_meter = metrics.Mean()
# loss_meter.update_state(float(loss))

acc_meter = metrics.accuracy()


