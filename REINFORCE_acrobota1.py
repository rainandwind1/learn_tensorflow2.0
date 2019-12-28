import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,Model
import numpy as np
import random
import collections
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'


# REINFOECR 算法 PolicyGradient(PG)

# 超参数
learning_rate = 0.002
gamma = 0.98
memory_size = 50000
max_length = 10000



env = gym.make('Acrobot-v1') 
action_choice = 3
state_num = 6
print(env.observation_space)
print(env.action_space)


class Policy(keras.Model):
    def __init__(self):
        super(Policy,self).__init__()
        self.fc1 = layers.Dense(256,kernel_initializer = 'he_normal')
        self.fc2 = layers.Dense(256,kernel_initializer = 'he_normal')
        self.fc3 = layers.Dense(action_choice,kernel_initializer = 'he_normal')
        self.memory = []#collections.deque(maxlen=max_length)

    def call(self,inputs,training = None):
        x = tf.constant(inputs,dtype = tf.float32)
        x = tf.expand_dims(x,axis = 0)
        x = tf.nn.relu(self.fc1(x))
        x = tf.nn.relu(self.fc2(x))
        x = self.fc3(x)
        return tf.nn.softmax(x,axis = 1) 


    def save_memory(self,target):
        self.memory.append(target)


def train(Policy,optimizer,tape):
    R = 0.
    for log_p, r in Policy.memory[::-1]:
        R = r + gamma*R
        loss = -log_p*R
        with tape.stop_recording():
            grads = tape.gradient(loss,Policy.trainable_variables)
            optimizer.apply_gradients(zip(grads,Policy.trainable_variables))
    Policy.memory = []



max_epoches = 10000
max_steps = 1000
o_interval = 20
score = 0.
P = Policy()
P(tf.ones((6,6)))
optimizer = optimizers.Adam(lr = learning_rate)
for epoch_i in range(max_epoches):
    s = env.reset()
    with tf.GradientTape() as tape:
        for step in range(max_steps):
            env.render()
            action_p = P(s)
            action = tf.random.categorical(tf.math.log(action_p),1)[0]
            action = int(action)
            obs,reward, done, info = env.step(action)
            P.save_memory((np.math.log(action_p[0][action]),reward))
            s = obs
            score += reward
            if epoch_i+1 % o_interval == 0:
                print("第%d回合得分：%f"%(epoch_i+1,score/o_interval))
                score = 0.
            if done:
                break
        train(P,optimizer,tape)
    del tape




