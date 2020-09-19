import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.arm import Arm
from src.GS import GS_Market
from src.player import Player


class Exp_example7():
    '''
        Experiments with the centralized UCB and centrailzed ETC algorithm with 20 players and 20 arms
    '''

    def __init__(self):
        self.num_players = 20
        self.num_arms = 20
        self.horizon = 8000
        self.trials = 50
        self.arms_var = 1
        self.reward_interval = 0.1
        self.two_side_market = GS_Market(self.num_players, self.num_arms)

    def run_example7_UCB(self, optimal=True):

        # initialize players
        players_ranking = np.arange(self.num_arms)
        players = [Player(self.num_arms, players_ranking)
                    for i in range(self.num_players)]
        
        # initialize arms
        arms_mean = np.linspace(0.9, 0, self.num_arms)
        arms_ranking = np.arange(self.num_players)
        arms = [Arm(self.num_players, arms_mean[i]*np.ones(self.num_players), self.arms_var, arms_ranking) 
                for i in range(self.num_arms)]

        # collect regrets
        self.regrets = np.zeros([self.num_players, self.horizon])

        # get optimal matching
        optimal_matching = self.two_side_market.get_optimal_matching(players, arms).tolist()
        
        # string = "optimal" if optimal else "pessimal"
        for _ in tqdm(range(self.trials), ascii=True, desc="Running the example 7 centralized UCB "):
            regrets_one_trial = [[] for i in range(self.num_players)]

            for _ in range(self.horizon):
                matching_result = self.two_side_market.match(players, arms)

                for a_idx in range(self.num_arms):
                    if matching_result[a_idx] != None:
                        p_idx = matching_result[a_idx]
                        regret = arms[optimal_matching.index(p_idx)].mean[p_idx] - \
                                 arms[a_idx].mean[p_idx]
                        regrets_one_trial[p_idx].append(regret)

            self.regrets += np.cumsum(np.array(regrets_one_trial), axis=1)
            self.two_side_market.reset(players)

        self.regrets /= self.trials

    def plot_example7(self):
        plt.figure(dpi=150)

        interval = 10
        plt.plot(self.regrets[0][0:-1:int(self.horizon/interval)],
                marker="o", color = 'blue', linewidth = 0.8, label = 'Agent 1')
        plt.plot(self.regrets[4][0:-1:int(self.horizon/interval)],
                marker="<", color = 'dodgerblue', linewidth = 0.8, label = 'Agent 5')
        plt.plot(self.regrets[9][0:-1:int(self.horizon/interval)],
                marker=">", color = 'lightskyblue', linewidth = 0.8, label = 'Agent 10')
        plt.plot(self.regrets[14][0:-1:int(self.horizon/interval)],
                marker="x", color = 'green', linewidth = 0.8, label = 'Agent 15')
        plt.plot(self.regrets[19][0:-1:int(self.horizon/interval)],
                marker="+", color = 'r', linewidth = 0.8, label = 'Agent 20')

        plt.ylabel('Expected Regret')
        plt.xlabel('Time')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    exp = Exp_example7()
    exp.run_example7_UCB()
    exp.plot_example7()
