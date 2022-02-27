from __future__ import division
import numpy as np

from rl.util import *
from rl.policy import Policy, GreedyQPolicy, EpsGreedyQPolicy

class MyEpsGreedyQPolicy(Policy):
    """Implement the epsilon greedy policy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """

    def __init__(self, eps=.1, masked_greedy=False):
        super(MyEpsGreedyQPolicy, self).__init__()
        self.eps = eps
        self.masked_greedy = masked_greedy

    def select_action(self, q_values, legal_actions):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            if legal_actions is None:
                action = np.random.random_integers(0, nb_actions - 1)
            else:
                action = int(np.random.choice(legal_actions, 1)[0])
        else:
            if legal_actions is None or not self.masked_greedy:
                action = np.argmax(q_values)
            else:
                for action in range(nb_actions):
                    if action not in legal_actions:
                        q_values[action] = -100000
                action = np.argmax(q_values)
        return action

    def get_config(self):
        """Return configurations of EpsGreedyPolicy

        # Returns
            Dict of config
        """
        config = super(MyEpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config


class MyGreedyQPolicy(Policy):
    """Implement the greedy policy

    Greedy policy returns the current best action according to q_values
    """

    def __init__(self, eps=.1, masked_greedy=False):
        super(MyGreedyQPolicy, self).__init__()
        self.masked_greedy = masked_greedy

    def select_action(self, q_values, legal_actions):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if legal_actions is None or not self.masked_greedy:
            action = np.argmax(q_values)
        else:
            for action in range(nb_actions):
                if action not in legal_actions:
                    q_values[action] = -100000
            action = np.argmax(q_values)

        return action

