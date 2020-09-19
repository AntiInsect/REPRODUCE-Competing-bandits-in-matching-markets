import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.arm import Arm
from src.GS import GS_Market
from src.player import Player


class Exp_example2():
    '''
        Experiments with the centralized UCB algorithm with 2 players and 2 arms
    '''
    
    def __init__(self):
        self.num_players = 2
        self.num_arms = 2
        self.arms_var = 1
        self.horizon = 400
        self.trials = 100
        self.freq = 20
        self.delta = np.linspace(0.0, 1.25, self.freq)
        self.two_side_market = GS_Market(self.num_players, self.num_arms)

    def run_example2_UCB(self, optimal=True):

        self.regret_example2 = np.zeros([self.num_players, len(self.delta)])

        # string = "optimal" if optimal else "pessimal"
        for i in tqdm(range(len(self.delta)), ascii=True, desc="Running the example 2 centralized UCB "):

            # initialize players
            true_player_rankings = [[0, 1], [1, 0]]
            players = [Player(self.num_arms, true_player_rankings[j])
                        for j in range(self.num_players)]

            # initialize arms
            arms_mean =[[self.delta[i], 0], [0, 1]]
            arms_rankings = [[0, 1], [0, 1]]
            arms = [Arm(self.num_players, arms_mean[j], self.arms_var, arms_rankings[j])
                    for j in range(self.num_arms)]

            # collect regrets
            regrets = np.zeros([self.num_players, self.horizon])

            # get the optimal matching
            optimal_matching = self.two_side_market.get_optimal_matching(players, arms).tolist()
            
            for _ in range(self.trials):
                regrets_one_trial = [[] for _ in range(self.num_players)]

                for _ in range(self.horizon):
                    matching_result = self.two_side_market.match(players, arms)

                    for a_idx in range(self.num_arms):
                        if matching_result[a_idx] != None:
                            p_idx = matching_result[a_idx]
                            regret = arms[optimal_matching.index(p_idx)].mean[p_idx] - \
                                     arms[a_idx].mean[p_idx]
                            regrets_one_trial[p_idx].append(regret)

                regrets += np.cumsum(np.array(regrets_one_trial), axis=1)
                self.two_side_market.reset(players)
        
            regrets /= self.trials
            self.regret_example2[:, i] = regrets[:, -1]


    def plot_example2(self):
        plt.figure(dpi=200)
        plt.plot(self.delta, self.regret_example2[0], marker="o", color = 'dodgerblue', linewidth = 0.8, label = 'Agent 1')
        plt.plot(self.delta, self.regret_example2[1], marker=">", color = 'blue', linewidth = 0.8, label = 'Agent 2')
        plt.ylabel('Expected Regret')
        plt.xlabel('Reward Gap of Agent 1')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    exp = Exp_example2()
    exp.run_example2_UCB()
    exp.plot_example2()
