import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,Model
import numpy as np
import random
import collections
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

# print(envs.registry.all())
env = gym.make('MountainCarContinuous-v0')

print(env.action_space)
print(env.observation_space)


for i in range(100):
    env.reset()
    for t in range(600):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break
        
env.close()

