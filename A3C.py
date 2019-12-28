import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses
import numpy as np
import collections
import gym
import copy
import threading
import multiprocessing
# A3C
env = gym.make('CartPole-v1')

class ActorCritic(keras.Model):

    def __init__(self,state_size,action_size):
        super(ActorCritic,self).__init__()
        self.state_size = state_size # 状态向量长度
        self.action_size = action_size # 动作数量
        # 策略网络Actor
        self.dense1 = layers.Dense(128,activation = 'relu')
        self.policy_logits = layers.Dense(action_size)
        # V 网络 Critic
        self.dense2 = layers.Dense(128,activation = 'relu')
        self.values = layers.Dense(1)

    def call(self,inputs):
        # 获得策略分布Pi
        x = self.dense1(inputs)
        logits = self.policy_logits(x)

        # 获得V(s)
        v = self.dense2(inputs)
        values = self.values(v)

        return logits,values

class Worker(threading.Thread):
    global_episode = 0
    global_avg_return = 0
    def __init__(self,server,opt,result_queue,idx):
        super(Worker,self).__init__()
        self.result_queue = result_queue # 共享队列
        self.server = server     # 中央模型
        self.opt = opt # 中央优化器
        self.client = ActorCritic(4,2) # 线程私有网络
        self.worder_idx = idx # 线程id
        self.env = gym.make('CartPole-v0').unwrapped
        self.ep_loss = 0.0

    def run(self):
        total_step = 1
        mem = Memory()
        while Worker.global_episode < 400:
            current_state = self.env.reset()
            mem.clear()
            ep_reward = 0.
            ep_step = 0
            self.ep_loss = 0
            time_count = 0
            dine = False
            while not done:
                # 获得Pi，未经过softmax
                logits, _ = self.client(tf.constant(current_state[None,:],dtype = tf.float32))
                probs = tf.nn.softmax(logits)
                action = np.random.choice(2,p=probs.numpy()[0])
                new_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -1
                ep_reward += reward
                mem.store(current_state,action,reward)

                if time_count == 20 or done:

                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done,new_state,mem)
                    self.ep_loss += float(total_loss)
                    grads = tape.gradient(total_loss,self.client.trainable_weights)
                    self.opt.apply_gradients(zip(grads,self.server.trainable_weights))

                    self.client.set_weights(self.server.get_weights())
                    mem.clear()
                    time_count = 0
                    if done:
                        Worker.global_avg_return = \ record(Worker.global_episode,ep_reward,self.worder_idx,Worker.global_avg_return,
                                                            self.result_queue,self.ep_loss,ep_steps)
                        Worker.global_episode += 1
                ep_steps += 1
                time_count += 1
                current_state = new_state
                total_step += 1
            self.result_queue.put(None) #结束线程

    def compute_loss(self,done,new_state,memory,gamma = 0.99):
        if done:
            reward_sum = 0.
        else:
            reward_sum = self.client(tf.constant(new_state[None,:],dtype = tf.float32))[-1].numpy()[0]
        discounted_rewards = []
        for reward in memory.rewards[::-1]:
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        logits,values = self.client(tf.constant(np.vstack(memory.states),dtype = tf.float32))
        advantage = tf.constant(np.array(discounted_rewards)[:None],dtype = tf.float32) - values
        value_loss = advantage ** 2
        policy = tf.nn.softmax(logits)
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = memory.actions,logits = logits)
        policy_loss *= tf.stop_gradient(advantage)
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels = policy,logits = logits)
        policy_loss -= 0.01*entropy
        total_loss = tf.reduce_mean((0.5*value_loss + policy_loss))
        return total_loss

class Agent:
    def __init__(self):
        self.opt = optimizers.Adam(1e-3)
        self.server = ActorCritic(4,2)
        self.server(tf.random.normal((2,4)))

    def train(self):
        res_queue = Queue()
        workers = [Worker(self.server,self.opt,res_queue,i) for i in range(multiprocessing.cpu_count())]
        for i,worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()
        moving_average_rewards = []
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break
        [w.join() for w in workers] # 等待线程退出




