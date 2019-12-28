import collections
import random
import gym,os
import  numpy as np
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers,optimizers,losses
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

env = gym.make('Acrobot-v1')
action_choice = 3
state_size = 6
print(env.observation_space)
for i in range(10):
    print(env.action_space.sample())