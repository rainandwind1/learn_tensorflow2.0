import gym
import tensorflow as tf
from tensorflow import keras,losses
from tensorflow.keras import layers,Model,optimizers
import os
import  numpy as np

batch_size = 1000
class Actor(keras.Model):
    def __init__(self):
        super(Actor,self).__init__()
        self.fc1 = layers.Dense(100,kernel_initializer = 'he_normal')
        self.fc2 = layers.Dense(2,kernek_initializer = 'he_normal')

    def call(self,inputs):
        x = self.fc1(inputs)
        x = self.fc2(tf.nn.relu(x))
        x = tf.nn.softmax(x,axis = 1)
        return x

class Critic(keras.Model):
    def __init__(self):
        super(Critic,self).__init__()
        self.fc1 = layers.Dense(100, kernel_initializer='he_normal')
        self.fc2 = layers.Dense(1, kernek_initializer='he_normal')

    def __call__(self,inputs):
        x = tf.nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PPO():
    # PPO 算法主体
    def __init__(self):
        super(PPO,self).__init__()
        self.actor = Actor()
        self.critic = Critic()
        self.buffer = [] #数据缓冲池
        self.actor_optimizer = optimizers.Adam(1e-3)
        self.critic_optimizer = optimizers.Adam(3e-3)

    def select_action(self,s):
        s = tf.constant(s,dtype = tf.float32)
        s = tf.expand_dims(s,axis = 0)
        prob = self.actor(s)
        a = tf.random.categorical(tf.math.log(prob),1)[0]
        a = int(a)
        return a,float(prob[0][a])

    def store_transition(self,trans):
        self.buffer.append(trans)

    def optimize(self):
        # 优化网络主函数
        # 从缓存中取出样本数据，转换成Tensor
        state = tf.constant([t.state for t in self.buffer],dtype = tf.float32)
        action = tf.constant([t.action for t in self.buffer],dtype = tf.float32)
        action = tf.reshape(action,[-1,1])
        reward = [t.reward for t in self.buffer]
        old_action_prob = tf.constant([t.action_prob for t in self.buffer],dtype = tf.float32)
        old_action_prob = tf.reshape(old_action_prob,[-1,1])
        # 通过MC方法循环计算R(st)
        R = 0
        Rs = []
        for r in reward[::-1]:
            R = r + gamma*R
            Rs.insert(0,R)
        Rs = tf.constant(Rs,dtype = tf.float32)

        # 对缓冲池数据大致迭代10遍
        for _ in range(round(10*len(self.buffer)/batch_size)):
            index = np.random.choice(np.arange(len(self.buffer)),batch_size,replace=False)
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                v_target = tf.expand_dims(tf.gather(Rs,index,axis = 0),axis = 1)
                v = self.critic(tf.gather(state,index,axis = 0))
                delta = v_target - v
                advantage = tf.stop_gradient(delta)
                a = tf.gather(action,index,axis = 0)
                pi = self.actor(tf.gather(state,index,axis = 0))
                indices = tf.expand_dims(tf.range(a.shape[0]),axis = 1)
                indices = tf.concat([indices,a],axis = 1)
                pi_a = tf.gather_nd(pi,indices)
                pi_a = tf.expand_dims(pi_a,axis = 1)

                ratio = (pi_a / tf.gather(old_action_prob,index,axis = 0))
                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio,1 - epsilon,1 + epsilon)*advantage
                policy_loss = -tf.reduce_mean(tf.minimum(surr1,surr2))
                value_loss = losses.MSE(v_target,v)
            grads = tape1.gradient(policy_loss,self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads,self.actor.trainable_variables))
            grads = tape2.gradients(value_loss,self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads,self.critic.trainable_variables))
            self.buffer = []
def Transition():
    return 0
env = gym.make("CartPole-v1")
def main():
    agent = PPO()
    returns = [] # 统计总回报
    total = 0   # 一段时间内的平均回报
    for i_epoch in range(500): # 训练回合数
        state = env.reset()
        for t in range(500):    # 最大训练步数
            # 环境交互
            action，action_prob = agent.select_action(state)
            next_state,reward,done,info = env.step(action)
            # 构建样本并存储
            trans = Transition(state,action,action_prob,reward,next_state)
            agent.store_transition(trans)
            state = next_state
            if done:
                if len(agent.buffer) >= batch_size:
                    agent.optimize()
                break

