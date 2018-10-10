import copy
import random
from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics


# create class for agents
class Player:
    def __init__(self, cooperate, defect, num_players):
        self.cooperate = cooperate
        self.defect = defect
        self.win_total = 0
        self.adj_matrix = np.ones((1, num_players), dtype=float)
        self.adj_matrix[0][0] = 0
        self.history = [self.defect / (self.cooperate + self.defect)]
        self.weight_history = [np.ones((1, num_players-1), dtype=float)[0].tolist()]

    def choose_strategy(self):
        defect_ratio = self.defect / (self.cooperate + self.defect)
        if random.uniform(0, 1) <= defect_ratio:
            defect = True
        else:
            defect = False
        return defect

    def save_history(self):
        self.history.append(self.defect / (self.cooperate + self.defect))
        self.weight_history.append(list(np.delete(self.adj_matrix, 0)))

    def update(self, defect, round_winnings, index):
        if defect:
            self.defect += round_winnings
            #self.history.append(self.defect / (self.cooperate + self.defect))
        else:
            self.cooperate += round_winnings
            #self.history.append(self.defect / (self.cooperate + self.defect))
        self.win_total += round_winnings
        self.adj_matrix[0][index] += round_winnings

    def get_adj_matrix(self):
        return self.adj_matrix

# TOO SLOW!!!!!!! make switches to list comprehensions
def run_game_network(payoffs, rounds, num_players, num_saves=0, async=True):
    # Selects rounds to save history
    if num_saves != 0:
        rounds_to_save = np.logspace(0, math.log10(rounds), num_saves)
        rounds_to_save = list(map(lambda z: math.floor(z), rounds_to_save))
        rounds_to_save[len(rounds_to_save)-1] -= 1
    # initialize agents
    agents = []
    for player in range(num_players):
        agents.append(Player(1, 1, num_players))
    # begin game
    for iteration in range(rounds):
        # make copy of agent list for synchronous
        agents_sync = copy.deepcopy(agents)
        # randomize order of agents
        random_order = random.sample(range(len(agents)), len(agents))
        # each agent chooses an opponent in each round
        for index in random_order:
            adj_matrix = agents[index].get_adj_matrix()
            random_num = random.uniform(0, sum(adj_matrix[0]))
            for opponent_index, item in enumerate(accumulate(adj_matrix[0])):
                if random_num <= item:
                    break
            # agents pick strategy and then update based on game results
            opponent_defect = agents[opponent_index].choose_strategy()
            agent_defect = agents[index].choose_strategy()
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
            for agent in agents:  # could use map() to save time
                agent.save_history()
        else:
            if iteration in rounds_to_save:
                # should add some way of saving iteration # to be on x axis
                for agent in agents:  # could use map() to save time
                    agent.save_history()
    return agents


# Game Setup and execution
payoff = [[(2/3), 0], [1, (1/3)]]
agent_results = run_game_network(payoff, 1000, 10, 30)


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

#loop_plot_defect_ratio(agent_results, 9)


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


#loop_plot_weights_ratio(agent_results, 7)


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
print(x, y, a, b)
exit()