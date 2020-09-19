import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


class GS_Market(object):
    '''
        Simulator of two side matching market with the GS algorithm
    '''

    def __init__(self, num_players, num_arms):
        self.num_players = num_players
        self.num_arms = num_arms
        self.t = 0
        self.players_rankings = []
        self.arms_rankings = []

    def proceed(self):
        self.t += 1

    def reset(self, players):
        self.t = 0
        for i in range(self.num_players):
            players[i].reset()

    def match(self, players, arms, ucb=True):
        '''
            The return is the matching result indexed by arms and valued by players.
            For now, we assume that the number of players and arms are the same
            such that there will always be an arm for each player and vice versa
        '''

        # enter a new round    
        self.proceed()

        # get the proposals from both sides
        self.players_rankings = np.array([players[j].get_ranking(ucb=ucb) \
                                            for j in range(self.num_players)], int) \
                                                .reshape(self.num_players, self.num_arms)
        self.arms_rankings = np.array([arms[j].get_ranking() \
                                        for j in range(self.num_arms)], int) \
                                            .reshape(self.num_arms, self.num_players)

        # get the final result of matching        
        matching = self.Gale_Shapley()

        # update the ucb estimation
        if ucb:
            for a_idx, p_idx in enumerate(matching):
                reward = arms[a_idx].sample(p_idx)
                players[p_idx].update(a_idx, reward, self.t)

        return matching


    def get_optimal_matching(self, players, arms):
        '''
            The optimal for each player is obtained using GS with true rankings from both sides
        '''
        self.players_rankings = np.array([players[j].get_true_ranking() \
                                            for j in range(self.num_players)], int) \
                                                .reshape(self.num_players, self.num_arms)
        self.arms_rankings = np.array([arms[j].get_ranking() \
                                        for j in range(self.num_arms)], int) \
                                            .reshape(self.num_arms, self.num_players)
        return self.Gale_Shapley()
    

    def Gale_Shapley(self):
        # propose_order records the order players should follow while proposing
        init_propose_order = np.zeros(self.num_players, int)
        propose_order = init_propose_order
        # matched record whether a specific player is matched or not
        matched = np.zeros(self.num_players, bool)
        # matching records the choice of a player for a specific arm
        matching = [[] for _ in range(self.num_arms)]

        # Terminates if all matched
        while np.sum(matched) != self.num_players:

            # plasyers propose at the same time
            for p_idx in range(self.num_players):
                if not matched[p_idx]:
                    p_proposal = self.players_rankings[p_idx][propose_order[p_idx]]
                    matching[p_proposal].append(p_idx)

            # arms choose its player
            for a_idx in range(self.num_arms):
                a_choices = matching[a_idx]

                if len(a_choices) != 0:    
                    # each arm chooses the its most preferable one
                    a_choice = next((x for x in self.arms_rankings[a_idx] if x in matching[a_idx]), None)
                    # update arm's choice where there should only be one left
                    matching[a_idx] = [a_choice]
                    # update player's state of matched
                    for p_idx in a_choices:
                        matched[p_idx] = (p_idx == a_choice)
                        propose_order[p_idx] += (1 - (p_idx == a_choice))
    
        return np.squeeze(matching)

    def attempt(self):
        
        pass
