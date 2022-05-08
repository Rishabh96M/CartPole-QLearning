# AgentLearning
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Agent Learning Class definition

import random
import numpy as np


class AgentLearning(object):
    """
    Definition
    ---
    Agent that can learn via Q-learning
    """

    def __init__(self, env, alpha, eps, gamma):
        self.env = env
        self.alpha = alpha
        self.eps = eps
        self.gamma = gamma
        self.Q_table = dict()
        self.training_trials = 0
        self.testing_trials = 0

        self.cart_position_bins = np.linspace(-2.4, 2.4, 11)[1:-1]
        self.pole_angle_bins = np.linspace(-2, 2, 11)[1:-1]
        self.cart_velocity_bins = np.linspace(-1, 1, 11)[1:-1]
        self.angle_rate_bins = np.linspace(-3.5, 3.5, 11)[1:-1]

    def build_state(self, features):
        """
        Definition
        ---
        Build state by concatenating features (bins) into 4 digit int
        """
        return int("".join(map(lambda feature: str(int(feature)), features)))

    def create_state(self, obs):
        """
        Definition
        ---
        Create state variable from observation

        Parameters
        ---
        obs: Observation list with format [horizontal position, velocity, angle
             of pole, angular velocity]

        Returns
        ---
        state: State tuple
        """
        state = self.build_state([np.digitize(x=[obs[0]],
                                              bins=self.cart_position_bins),
                                  np.digitize(x=[obs[1]],
                                              bins=self.pole_angle_bins),
                                  np.digitize(x=[obs[2]],
                                              bins=self.cart_velocity_bins),
                                  np.digitize(x=[obs[3]],
                                  bins=self.angle_rate_bins)])
        return state

    def choose_action(self, state):
        """
        Definition
        ---
        Given a state, choose an action.

        Parameters
        ---
        state: State of the agent.

        Returns
        ---
        action: Action that agent will take.
        """
        if random.random() < self.eps:
            action = self.env.action_space.sample()
        else:
            # Find max Q value
            max_Q = self.get_maxQ(state)
            actions = []
            for key, value in self.Q_table[state].items():
                if value == max_Q:
                    actions.append(key)
            if len(actions) != 0:
                action = random.choice(actions)
        return action

    def create_Q(self, state, valid_actions):
        """
        Definition
        ---
        Update the Q table given a new state/action pair.

        Parameters
        ---
        state: List of state booleans.
        valid_actions: List of valid actions for environment.
        """
        if state not in self.Q_table:
            self.Q_table[state] = dict()
            for action in valid_actions:
                self.Q_table[state][action] = 0.0
        return

    def get_maxQ(self, state):
        """
        Definition
        ---
        Find the maximum Q value in a given Q table.

        Parameters
        ---
        Q_table: Q table dictionary.
        state: List of state booleans.

        Returns
        ---
        maxQ: Maximum Q value for a given state.
        """
        maxQ = max(self.Q_table[state].values())
        return maxQ

    def learn(self, state, action, prev_reward, prev_state, prev_action):
        """
        Definition
        ---
        Update the Q-values

        Parameters
        ---
        state: State at current time step.
        action: Action at current time step.
        prev_reward: Reward at previous time step.
        prev_state: State at previous time step.
        prev_action: Action at previous time step.
        """
        self.Q_table[prev_state][prev_action] = (1 - self.alpha) * \
            self.Q_table[prev_state][prev_action] + self.alpha * \
            (prev_reward + (self.gamma * self.get_maxQ(state)))
        return
