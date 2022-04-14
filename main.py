# CartPole-QLearning
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Progam to balance the pole on a cart using Q Learning and openAI
#              CartPole-V1 environment.

import gym
import numpy as np
import random
import math

env = gym.make('CartPole-v1')
env.reset()

for d in range(100):
    env.render()
    env.step(env.action_space.sample())
