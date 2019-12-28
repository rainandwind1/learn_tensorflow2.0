import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import copy


dataset_path = keras.utils.get_file("auto-mpg.data","http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/autompg.data")

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(dataset_path, names=column_names,na_values = "?", comment='\t',
sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()

dataset.isna().sum() #统计空白数据
dataset = dataset.dropna()#删除空白数据
dataset.isna().sum()#再次统计空白数据

origin = dataset.pop('origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

train_dataset = dataset.sample(frac = 0.8,random_state= 0)
test_dataset = dataset.drop(train_dataset.index)
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

print(normed_train_data.shape,train_labels.shape)
print(normed_test_data.shape,test_labels.shape)


train_db = tf.data.Dataset.from_sparse_tensor_slices((normed_train_data.values,train_labels.values))
train_db = train_db.shuffle(100).batch(32)

# 创建网络
class Network(keras.Model):
    #回归网络
    def __init__(self):
        super(Network,self).__init__()
        #创建3个全连接层
        self.fc1 = layers.Dense(64,activation = 'relu')
        self.fc2 = layers.Dense(64,activation = 'relu')
        self.fc3 = layers.Dense(1)

    def call(self,inputs,training = None,mask = None):
        #依次通过三个全连接层
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

#训练与测试
model = Network()
model.build(input_shape=(4,9))
model.summary()
optimizer = tf.keras.optimizers.RMSprop(0.001)

#网络训练部分
for epoch in range(200):
    for step,(x,y) in enumerate(train_db):
        #梯度记录器
        with tf.GradientTape() as tape:
            out = model(x)
            loss = tf.reduce_mean(keras.losses.MSE(y,out))
            mae_loss = tf.reduce_mean(keras.losses.MAE(y,out))
            if step%10 == 0:
                print(epoch,step,float(loss))

            grads = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
            
