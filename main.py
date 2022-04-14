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

    # Dummy code to check if environment is working
    for d in range(100):
        env.render()
        # take a random action
        env.step(env.action_space.sample())
    env.close()

    # Initialising AgentLearning
    agent = AgentLearning(env, 0.9, 1.0, 0.9)
    print(agent)

    '''
    Yet to implement Q Learning Policy
    All support methods are implemented in Agent Learning
    '''
