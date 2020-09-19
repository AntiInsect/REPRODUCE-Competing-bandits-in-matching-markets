import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.arm import Arm
from src.GS import GS_Market
from src.player import Player


class Exp_centralized_UCB_ETC(object):
    '''
        Extra experiments with the centralized UCB and centralized ETC algorithm
    '''
    
    def __init__(self):
        self.num_players = 10
        self.num_arms = 10
        self.arms_var = 1
        self.horizon = 4000
        self.trials = 10

        self.players_rankings = np.arange(self.num_arms)
        self.players = [Player(self.num_arms, self.players_rankings) 
                        for j in range(self.num_players)]

        self.arms_mean = np.linspace(0.9, 0, self.num_arms)
        self.arms_rankings = np.arange(self.num_players)
        self.arms = [Arm(self.num_players, self.arms_mean[j]*np.ones(self.num_players), self.arms_var, self.arms_rankings) 
                        for j in range(self.num_arms)]
        
        self.two_side_market = GS_Market(self.num_players, self.num_arms)
        self.optimal_matching = self.two_side_market.get_optimal_matching(self.players, self.arms).tolist()

        self.regrets_ucb = []
        self.regrets_etc = []


    def run_centralized_UCB(self, optimal=True):

        regrets = np.zeros([self.num_players, self.horizon])
        
        # string = "optimal" if optimal else "pessimal"
        for _ in tqdm(range(self.trials), ascii=True, desc="Running the centralized UCB "):
            regrets_one_trial = [[] for _ in range(self.num_players)]

            for _ in range(self.horizon):
                matching_result = self.two_side_market.match(self.players, self.arms)
                
                for a_idx in range(self.num_arms):
                    if matching_result[a_idx] != None:
                        p_idx = matching_result[a_idx]
                        regret = self.arms[self.optimal_matching.index(p_idx)].mean[p_idx] - \
                                 self.arms[a_idx].mean[p_idx]
                        regrets_one_trial[p_idx].append(regret)
        
            regrets += np.cumsum(np.array(regrets_one_trial), axis=1)
            self.two_side_market.reset(self.players)
        
        regrets /= self.trials
        return regrets

    def run_centralized_ETC(self, h, optimal=True):
        regrets = np.zeros([self.num_players, self.horizon])
        
        # string = "optimal" if optimal else "pessimal"
        for _ in tqdm(range(self.trials), ascii=True, desc="Running the centralized ETC "):
            regrets_one_trial = [[] for _ in range(self.num_players)]

            matching_result = None
            for t in range(self.horizon):
                # Explore
                if t < h * self.num_arms:
                    for p_idx in range(self.num_players):
                        a_idx = ((t + p_idx) % self.num_arms)
                        reward = self.arms[a_idx].sample(p_idx)

                        self.two_side_market.proceed()
                        self.players[p_idx].update(a_idx, reward, self.two_side_market.t)
                        regret = self.arms[self.optimal_matching.index(p_idx)].mean[p_idx] - \
                                 self.arms[a_idx].mean[p_idx]
                        regrets_one_trial[p_idx].append(regret)
                
                # Commit
                else:
                    if t == h * self.num_arms:
                        matching_result = self.two_side_market.match(self.players, self.arms, ucb=False)
                    
                    # self.two_side_market.proceed()
                    for a_idx in range(self.num_arms):
                        if matching_result[a_idx] != None:
                            p_idx = matching_result[a_idx]
                            regret = self.arms[self.optimal_matching.index(p_idx)].mean[p_idx] - \
                                    self.arms[a_idx].mean[p_idx]
                            regrets_one_trial[p_idx].append(regret)
            
            regrets += np.cumsum(np.array(regrets_one_trial), axis=1)
            self.two_side_market.reset(self.players)
        
        regrets /= self.trials
        return regrets

    def run_centralized_UCB_ETC(self, explore_rounds):
        self.regrets_ucb = self.run_centralized_UCB()
        self.regrets_etc = [self.run_centralized_ETC(h) for h in explore_rounds]

    def plot_centralized_UCB_ETC(self, explore_rounds):
        plt.figure(dpi = 200)

        plt.plot(np.linspace(0, self.horizon, 10),
                 self.regrets_ucb[0][0:-1:int(self.horizon/10)], 
                 marker="o", linewidth = 0.8, label = 'UCB-Optimal')

        for i, h in enumerate(explore_rounds):
            plt.plot(self.regrets_etc[i][0], linewidth = 0.8, label = 'ETC, h = ' + str(h))
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Expected Regret')
        plt.title("Optimal Regret of Agent 1")
        plt.show()

if __name__ == "__main__":

    exp = Exp_centralized_UCB_ETC()
    explore_rounds = [25, 50, 100, 200]
    exp.run_centralized_UCB_ETC(explore_rounds)
    exp.plot_centralized_UCB_ETC(explore_rounds)
