################################################
############ Issues to be resolved #############
################################################
# change x axis to round of save
# save weights to history as ratio (proportion of interactions requires ratio, currently raw values)
# Make opponent also make strategy/opponent errors
# In analysis why more time steps in link weigh ratio plot

import copy
import random
from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics
import csv

# Set Discount and Error Rates
discount = 0.99
error = 0.01


# create class for agents
class Player:
    def __init__(self, cooperate, defect, num_players, self_index):
        self.self_index = self_index
        self.cooperate = cooperate
        self.defect = defect
        self.current_defect_ratio = self.defect / (self.cooperate + self.defect)
        self.win_total = 0
        self.adj_matrix = [1] * num_players
        self.adj_matrix[self_index] = 0
        self.error_adj_matrix = self.adj_matrix  # play uniform strategy if error
        self.history = [self.current_defect_ratio]
        self.weight_history = [list(self.adj_matrix)]

    def choose_strategy(self, choice_error):
        if choice_error:
            defect_ratio = 0.5
        else:
            defect_ratio = self.current_defect_ratio
        if random.uniform(0, 1) <= defect_ratio:
            defect = True
        else:
            defect = False
        return defect

    def save_history(self):
        self.history.append(self.current_defect_ratio)
        self.weight_history.append(list(self.adj_matrix))

    def update(self, defect, round_winnings, index):
        if defect:
            self.defect = discount*self.defect + round_winnings
        else:
            self.cooperate = discount*self.cooperate + round_winnings
        self.current_defect_ratio = self.defect / (self.cooperate + self.defect)
        self.win_total += round_winnings
        self.adj_matrix[index] = discount*self.adj_matrix[index] + round_winnings

    def get_adj_matrix(self):
        return self.adj_matrix

    def get_error_adj_matrix(self):
        return self.error_adj_matrix


def calc_proportion_of_interactions(agents):
    cum_prob = {"DD": 0, "DC": 0, "CD": 0, "CC": 0}
    interaction_types = cum_prob.keys()
    for agent in agents:
        for neighbor_index in range(len(agents)):
            for interaction in interaction_types:
                if interaction == 'DD':
                    cum_prob[interaction] = cum_prob[interaction] + (agent.current_defect_ratio * \
                                            agent.adj_matrix[neighbor_index] * \
                                            agents[neighbor_index].current_defect_ratio) / len(agents)
                elif interaction == 'DC':
                    cum_prob[interaction] = cum_prob[interaction] + (agent.current_defect_ratio * \
                                            agent.adj_matrix[neighbor_index] * \
                                            (1 - agents[neighbor_index].current_defect_ratio)) / len(agents)
                elif interaction == 'CD':
                    cum_prob[interaction] = cum_prob[interaction] + ((1 - agent.current_defect_ratio) * \
                                            agent.adj_matrix[neighbor_index] * \
                                            agents[neighbor_index].current_defect_ratio) / len(agents)
                else:
                    cum_prob[interaction] = cum_prob[interaction] + ((1 - agent.current_defect_ratio) * \
                                            agent.adj_matrix[neighbor_index] * \
                                            (1 - agents[neighbor_index].current_defect_ratio)) / len(agents)
    # convert dict to normalized list of values to return???? WHY THOUGH??????
    norm_cum_prob = [float(i) / sum(list(cum_prob.values())) for i in list(cum_prob.values())]
    return norm_cum_prob


# TOO SLOW!!!!!!! make switches to list comprehensions
def run_game_network(payoffs, rounds, num_players, num_saves=0, async=True):
    # Selects rounds to save history (will save less than request unless iterations >> saves)
    if num_saves != 0:
        rounds_to_save = np.logspace(0, math.log10(rounds), num_saves)
        rounds_to_save = list(map(lambda z: math.floor(z), rounds_to_save))
        rounds_to_save[len(rounds_to_save)-1] -= 1
    # initialize agents
    agents = []
    for player in range(num_players):
        agents.append(Player(1, 1, num_players, player))
    proportion_interaction_history = [calc_proportion_of_interactions(agents)]
    # begin game
    for iteration in range(rounds):
        # RVs to determine errors
        strategy_error = random.uniform(0, 1)
        opponent_error = random.uniform(0, 1)
        # make copy of agent list for synchronous
        agents_sync = copy.deepcopy(agents)
        # randomize order of agents
        random_order = random.sample(range(len(agents)), len(agents))
        # each agent chooses an opponent in each round
        for index in random_order:
            if opponent_error <= error:
                adj_matrix = agents[index].get_error_adj_matrix()
            else:
                adj_matrix = agents[index].get_adj_matrix()
            random_num = random.uniform(0, sum(adj_matrix))
            for opponent_index, item in enumerate(accumulate(adj_matrix)):
                if random_num <= item:
                    break
            # Check for strategy error
            choice_error = (strategy_error <= error)
            # agents pick strategy and then update based on game results
            opponent_defect = agents[opponent_index].choose_strategy(False)  # opponent never makes choice errors
            agent_defect = agents[index].choose_strategy(choice_error)
            agent_winnings = payoffs[agent_defect][opponent_defect]
            opponent_winnings = payoffs[opponent_defect][agent_defect]
            if async:
                agents[index].update(agent_defect, agent_winnings, opponent_index)
                agents[opponent_index].update(opponent_defect, opponent_winnings, index)
            else:  # need to check accuracy!!!!!!!
                agents_sync[index].update(agent_defect, agent_winnings, opponent_index)
                agents_sync[opponent_index].update(opponent_defect, opponent_winnings, index)
        if not async:
            agents = agents_sync
        # save players defect ratio at the end of each round
        if num_saves == 0:  # Default saves every round
            proportion_interaction_history.append(calc_proportion_of_interactions(agents))
            for agent in agents:  # could use map() to save time
                agent.save_history()
        else:
            if iteration in rounds_to_save:
                proportion_interaction_history.append(calc_proportion_of_interactions(agents))
                # should add some way of saving iteration # to be on x axis
                for agent in agents:  # could use map() to save time
                    agent.save_history()
    return agents, proportion_interaction_history


# ADD A WAY TO WRITE PROPORTION OF RESULTS TO FILE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def write_results(results, proportion_interaction_history ):
    with open('run_results.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(proportion_interaction_history )
        for result in results:
                writer.writerow(result.history)
                writer.writerow(result.weight_history)


# Game Setup and execution
payoff = [[(2/3), 0], [1, (1/3)]]
agent_results, proportion_history = run_game_network(payoff, 1000, 10, 50)
write_results(agent_results, proportion_history)
exit()


def loop_plot_defect_ratio(results, plot_number, num_columns=4):
    random_sample = random.sample(range(len(results)), plot_number)
    num_rows = math.ceil(plot_number/num_columns)
    x = []
    y = []
    for index in random_sample:
        y.append(results[index].history)
        x.append(list(range(len(results[index].history))))
    plots = zip(x, y)
    fig = plt.figure()
    # fig.suptitle('test title', fontsize=20)
    for index, values in enumerate(plots):
        ax = plt.subplot(num_rows, num_columns, index+1)
        ax.set_xscale('symlog')
        plt.title('Agent ' + str(random_sample[index]))
        plt.tight_layout()
        plt.xlabel('Round')
        plt.ylabel('Defection Ratio')
        plt.plot(values[0], values[1])
    plt.show()

loop_plot_defect_ratio(agent_results, 10)

def loop_plot_weights_ratio(results, plot_number, num_columns=4):
    random_sample = random.sample(range(len(results)), plot_number)
    num_rows = math.ceil(plot_number/num_columns)
    x = []
    y = []
    for index in random_sample:
        # convert to ratio of weights in each round
        for pos, value in enumerate(results[index].weight_history):
            new_list = []
            for item in value:
                new_list.append(item / sum(value))
                results[index].weight_history[pos] = new_list
        # convert lists by round to lists by agent
        by_agent = list(zip(*results[index].weight_history))
        y.append(by_agent)
        x.append(list(range(len(results[index].history))))
    plots = zip(x, y)
    plt.figure()
    for index, values in enumerate(plots):
        ax = plt.subplot(num_rows, num_columns, index+1)
        ax.set_xscale('symlog')
        plt.title('Agent ' + str(random_sample[index]))
        plt.tight_layout()
        plt.xlabel('Round')
        plt.ylabel('Opponent Weights Ratio')
        # plots the weights for each other player in the game
        for opponent in range(len(values[1])):
            plt.plot(values[0], values[1][opponent])
    plt.show()


loop_plot_weights_ratio(agent_results, 5)
exit()


def summary_statistics(results):
    bottom_quarter = 0
    top_quarter = 0
    ratios = []
    sd = []
    # mean defect ratio and sd of weights of all agents at end
    for result in results:
        ratios.append(result.history[-1])
        if result.history[-1] >= 0.75:
            top_quarter += 1
        elif result.history[-1] <= 0.25:
            bottom_quarter += 1
        sd.append(statistics.stdev(result.weight_history[-1]))
    return statistics.mean(ratios), statistics.mean(sd), bottom_quarter, top_quarter


x, y, a, b = summary_statistics(agent_results)
print('Mean Defect Ratio', x, y, a, b)