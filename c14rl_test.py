import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Model,optimizers
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

# 超参数
learning_rate = 0.001
gamma = 0.98

class Policy(keras.Model):
    # 策略网络，生成动作的概率分布
    def __init__(self):
        super(Policy, self).__init__()
        self.data = [] # 存储轨迹
        # 输入为长度为4的向量，输出为左、右2个动作
        self.fc1 = layers.Dense(256, kernel_initializer='he_normal')
        self.fc2 = layers.Dense(256, kernel_initializer='he_normal')
        self.fc3 = layers.Dense(3, kernel_initializer='he_normal')
        # 网络优化器
        self.optimizer = optimizers.Adam(lr=learning_rate)

    def call(self, inputs, training=None):
        # 状态输入s的shape为向量：[4]
        x = tf.nn.relu(self.fc1(inputs))
        x = tf.nn.relu(self.fc2(x))
        x = tf.nn.softmax(self.fc3(x), axis=1)
        return x

    def put_data(self, item):
        # 记录r,log_P(a|s)
        self.data.append(item)

    def train_net(self, tape):
        # 计算梯度并更新策略网络参数。tape为梯度记录器
        R = 0 # 终结状态的初始回报为0
        for r, log_prob in self.data[::-1]:#逆序取
            R = r + gamma * R # 计算每个时间戳上的回报
            # 每个时间戳都计算一次梯度
            # grad_R=-log_P*R*grad_theta
            loss = -log_prob * R
            with tape.stop_recording():
                # 优化策略网络
                grads = tape.gradient(loss, self.trainable_variables)
                # print(grads)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.data = [] # 清空轨迹


pi = Policy()
pi(tf.ones((6,6)))
score = 0.
env = gym.make("Acrobot-v1")
for n_epi in range(1000):
    s = env.reset()
    with tf.GradientTape(persistent=True) as tape:
        for t in range(1001):
            env.render()
            s = tf.constant(s,dtype=tf.float32)
            s = tf.expand_dims(s,axis = 0)
            prob = pi(s)
            a = tf.random.categorical(tf.math.log(prob), 1)[0]
            a = int(a)
            s_prime,r,done,info = env.step(a)
            pi.put_data((r,tf.math.log(prob[0][a])))
            s = s_prime
            score += r
            if n_epi+1 % 20 == 0:
                print("第%d回合得分：%f"%(n_epi+1,score/20))
                score = 0.
            if done:
                #print("第%d回合得分为%d\n"%(n_epi,score))
                break
        pi.train_net(tape)
    del tape
env.close()

