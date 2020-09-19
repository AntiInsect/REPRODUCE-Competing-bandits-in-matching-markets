import numpy as np
from random import shuffle


class Player(object):

    def __init__(self, num_arms, true_ranking, oracle=False):
        self.num_arms = num_arms
        self.true_ranking = true_ranking
        self.oracle = oracle    
        self.epsilon = 10**(-10)
        self.reset()    

    def reset(self):
        self.count = np.zeros(self.num_arms)
        self.est_mean = np.zeros(self.num_arms)
        self.ucb = np.ones(self.num_arms) * np.inf

    def update(self, a_idx, reward, t):
        self.count[a_idx] += 1
        self.est_mean[a_idx] += (reward-self.est_mean[a_idx]) / self.count[a_idx]
        self.ucb[a_idx] = self.est_mean[a_idx] + np.sqrt(3 * np.log(t) / (2*(self.count[a_idx] + self.epsilon)))

    def get_true_ranking(self):
        return self.true_ranking

    def get_ranking(self, ucb=True):
        if self.oracle:
            return self.get_true_ranking()

        if ucb:
            return np.argsort(-self.ucb)
        else:
            return np.argsort(-self.est_mean)
