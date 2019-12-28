import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,Model
import numpy as np
import random
import collections
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'


# DQN   （1）经验回放   （2）固定target目标
# Acrobot-v1
# env = gym.make('CartPole-v1')#('Acrobot-v1') MountainCarContinuous-v0
env = gym.make('Acrobot-v1')
action_choice = 3
state_size = 6
print(env.observation_space)
print(env.action_space)
# print(env.observation_space.high,env.observation_space.low)


# 超参数设置
learning_rate = 0.01
gamma = 0.98
memory_length = 50000
memory_batch = 32
time_apply = 20  # 每多少步更新一次target网络的参数
# none alpha = 1

class DQN(keras.Model):
    def __init__(self):
        super(DQN,self).__init__()
        self.fc1 = layers.Dense(512,kernel_initializer = 'he_normal') # 正态分布 初始化权重
        self.fc2 = layers.Dense(256,kernel_initializer = 'he_normal') 
        self.fc3 = layers.Dense(256,kernel_initializer = 'he_normal') 
        self.fc4 = layers.Dense(action_choice,kernel_initializer = 'he_normal') 
        # self.memory = {'s':[],'a':[],'r':[]}

    def call(self,inputs,training = None):
        x = tf.nn.relu(self.fc1(inputs))
        x = tf.nn.relu(self.fc2(x))
        x = tf.nn.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    # def memory_list(self,state,action,reward):
    #     self.memory['s'].append(state)
    #     self.memory['a'].append(action)
    #     self.memory['r'].append(reward)
    #     return self.memory

    def choose_action(self,state,epsilon):
        state = tf.constant(state,dtype = tf.float32)
        state = tf.expand_dims(state,axis = 0)
        x = self(state)[0]
        p = np.random.rand()
        if p < epsilon:
            action = env.action_space.sample()
        else:
            action = int(tf.argmax(x))
        return action


class Replay_memory():
    def __init__(self):
        self.memory = collections.deque(maxlen=memory_length)

    def add_memory(self,bag):
        self.memory.append(bag)

    def sample_train_batch(self,sample_number):
        s,a,r,s_next,done_flag = [],[],[],[],[]
        sample_batch = random.sample(self.memory,sample_number)
        for sample in sample_batch:
            s_,a_,r_,s_next_,done_flag_ = sample
            s.append(s_)
            a.append([a_])
            r.append([r_])
            s_next.append(s_next_)
            done_flag.append([done_flag_])
        
        return tf.constant(s,dtype = tf.float32),\
        tf.constant(a,dtype = tf.int32),\
        tf.constant(r,dtype = tf.float32),\
        tf.constant(s_next,dtype = tf.float32),\
        tf.constant(done_flag,dtype = tf.float32)

    def size(self):
        return len(self.memory)

def train_net(q_net,q_target,memory_list,optimizer):
    huber = losses.Huber()  # Huber loss  一个分段误差函数
    for i in range(10):    # 每次回放10次 batch_size
        s,a,r,s_next,done_flag = memory_list.sample_train_batch(memory_batch)
        with tf.GradientTape() as tape:
            qa_out = q_net(s)
            a_index = tf.expand_dims(tf.range(a.shape[0]),axis = 1)
            a_index = tf.concat([a_index,a],axis = 1)
            q_a = tf.gather_nd(qa_out,a_index)
            q_a = tf.expand_dims(q_a,axis = 1)
            q_tmax =  tf.reduce_max(q_target(s_next),axis = 1,keepdims = True)
            q_t = r + gamma * q_tmax * done_flag
            loss = huber(q_a,q_t)
        grads = tape.gradient(loss,q_net.trainable_variables)
        optimizer.apply_gradients(zip(grads,q_net.trainable_variables))



        




q_net = DQN()    # q网络
q_target = DQN() # 固定更新的target网络
# q_net.build(input_shape = (2,4))     # CartPole-v1
# q_target.build(input_shape = (2,4))
q_net.build(input_shape = (1,state_size))
q_target.build(input_shape = (1,state_size))
for src, dest in zip(q_net.variables, q_target.variables):
        dest.assign(src) # 影子网络权值来自Q 保证初始权值相同
memory_list = Replay_memory() # 经验回放
optimizer = optimizers.Adam(lr = learning_rate)
max_epoches = 10000
max_steps = 1000
score = 0
for epoch_i in range(max_epoches):
    s = env.reset()
    epsilon = max(0.01,0.2-0.01*(epoch_i)/200)
    for step in range(max_steps):
        env.render()
        action = q_net.choose_action(s,epsilon)
        obs,reward,done,info = env.step(action)
        done_mask = 0.0 if done else 1.0
        memory_list.add_memory((s,action,reward/10.,obs,done_mask))          # 当前状态state s 采取的动作 action 该step获得的奖励 reward
        s = obs
        score += reward
        if done:
            if (epoch_i+1)%20 == 0:
                print("%d回合得分：%d"%(epoch_i + 1,score/20))
                score = 0
            break
    if memory_list.size() >= 2000:
        train_net(q_net,q_target,memory_list,optimizer)
    if (epoch_i+1)%time_apply == 0:
        for src,dest in zip(q_net.variables,q_target.variables):
            dest.assign(src)

env.close()




# env.reset()
# for epoch in range(200):
#     for step in range(200):
#         env.render()
#         a = env.action_space.sample()
#         observation,reward,done,info = env.step(a)
#         if done:
#             break

# env.close()

