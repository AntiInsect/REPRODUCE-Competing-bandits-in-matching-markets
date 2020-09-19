import numpy as np


class Arm(object):
    
    def __init__(self, num_players, mean, var, ranking):
        self.num_players = num_players
        self.mean = mean
        self.var = var
        
        # for arm, the "ranking" is out of the "target reference"
        self.ranking = ranking

    def sample(self, p_idx):
        return np.random.normal(self.mean[p_idx], self.var)
    
    def get_ranking(self):
        return self.ranking
