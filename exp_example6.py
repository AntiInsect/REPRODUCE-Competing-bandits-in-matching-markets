import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.arm import Arm
from src.GS import GS_Market
from src.player import Player


class GS_Market_(GS_Market):
    def __init__(self, num_players, num_arms):
        super().__init__(num_players, num_arms)
    
    def reset(self, players):
        super().reset(players)
        players[self.num_players-1].ucb = np.array([2.3, 0, 0])


class Exp_example6():

    def __init__(self):
        self.num_players = 3
        self.num_arms = 3
        self.horizon = 200
        self.trials = 10
        self.arms_var = 1
        self.two_side_market = GS_Market_(self.num_players, self.num_arms)

    def run_example6_UCB(self, optimal=True):

        # initialize players
        players_rankings = [[0, 1, 2], [1, 0, 2], [2, 0, 1]]
        players = [Player(self.num_arms, players_rankings[i], oracle=True) for i in range(self.num_players-1)]
        players.append(Player(self.num_arms, players_rankings[self.num_players-1]))
        # players[self.num_players-1].ucb = np.array([2.3, 0, 0])
        
        # initialize arms
        arms_mean =[[2, 1, 1.95], [1, 2, 0],[0, 0, 2]]
        arms_rankings = [[1, 2, 0], [0, 1, 2], [2, 0, 1]]
        arms = [Arm(self.num_players, arms_mean[i], self.arms_var, arms_rankings[i]) 
                for i in range(self.num_arms)]

        # collect regrets
        self.regrets = np.zeros([self.num_players, self.horizon])

        # get the optimal matching
        optimal_matching = self.two_side_market.get_optimal_matching(players, arms).tolist()
        
        string = "optimal" if optimal else "pessimal"
        for _ in tqdm(range(self.trials), ascii=True, desc="Running the example 6 centralized UCB "+string):
        
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

    def plot_example6(self):
        plt.figure(dpi=150)

        delta = np.linspace(0, self.horizon, 10)
        plt.plot(delta, self.regrets[0][0:-1:int(self.horizon/10)],
                marker="o", color = 'blue', linewidth = 0.8, label = 'Agent 1')
        plt.plot(delta, self.regrets[1][0:-1:int(self.horizon/10)],
                marker="<", color = 'dodgerblue', linewidth = 0.8, label = 'Agent 2')
        plt.plot(delta, self.regrets[2][0:-1:int(self.horizon/10)],
                marker=">", color = 'lightskyblue', linewidth = 0.8, label = 'Agent 3')

        plt.ylabel('Expected Regret')
        plt.xlabel('Time')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    exp = Exp_example6()
    exp.run_example6_UCB()
    exp.plot_example6()
