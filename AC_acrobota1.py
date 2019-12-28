import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,Model
import numpy as np
import random
import collections
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'


# Actor Critic 算法
env = gym.make('Acrobot-v1') 




env.close()



