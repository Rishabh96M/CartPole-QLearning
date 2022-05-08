# CartPole-QLearning
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Progam to balance the pole on a cart using Q Learning and openAI
#              CartPole-V1 environment.

import gym
from AgentLearning import AgentLearning
import stats


def q_learning(env, agent):
    '''
    Definition
    ---
    Q-learning policy definition

    Parameters
    ---
    env: Gym enviroment object
    agent: Learning agent

    Returns
    ---
    training_totals: Rewards for training
    testing_totals: Rewards for testing
    history: list of epsilon and alpha values.
    '''
    valid_actions = [0, 1]
    tolerance = 0.001
    training = True
    training_totals = []
    testing_totals = []
    history = {'epsilon': [], 'alpha': []}
    prev_state = 0
    prev_action = 0
    prev_reward = 0

    for episode in range(800):
        episode_rewards = 0
        obs = env.reset()

        if agent.eps < tolerance:
            agent.alpha = 0
            agent.eps = 0
            training = False

        agent.eps = agent.eps * 0.99
        for step in range(200):
            env.render()
            state = agent.create_state(obs)
            agent.create_Q(state, valid_actions)
            action = agent.choose_action(state)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward

            if step != 0:
                agent.learn(state, action, prev_reward,
                            prev_state, prev_action)
            prev_state = state
            prev_action = action
            prev_reward = reward
            if done:
                break

        print('\nEpisode: ', episode)
        print('Episode Rewards: ', episode_rewards)
        if training:
            training_totals.append(episode_rewards)
            agent.training_trials += 1
            history['epsilon'].append(agent.eps)
            history['alpha'].append(agent.alpha)
        else:
            testing_totals.append(episode_rewards)
            agent.testing_trials += 1
            if agent.testing_trials == 100:
                break
    return training_totals, testing_totals, history


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    env.reset(seed=21)
    agent = AgentLearning(env, 0.9, 1.0, 0.9)
    training_totals, testing_totals, history = q_learning(env, agent)

    stats.display_stats(agent, training_totals, testing_totals, history)
    stats.save_info(agent, training_totals, testing_totals)

    print("Environment SOLVED!!!")
