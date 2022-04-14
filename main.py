# CartPole-QLearning
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Progam to balance the pole on a cart using Q Learning and openAI
#              CartPole-V1 environment.

import gym
from AgentLearning import AgentLearning


if __name__ == '__main__':
    # Initialising openAI gym environment of CartPole-v1
    env = gym.make('CartPole-v1')

    # Resetting the environment
    env.reset()

    agent = AgentLearning(env, 0.9, 1.0, 0.9)
    print(agent)

    # Dummy code to check if environment is working
    for d in range(100):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()

    '''
    Yet to implement Q Learning Policy
    All support methods are implemented in Agent Learning
    '''
