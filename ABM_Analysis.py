import pandas as pd
import matplotlib.pyplot as plt
import ast

full_data = pd.read_csv('run_results.csv', header=None)
proportion_data = full_data[full_data.index == 0]
full_data = full_data.drop([0])  # Dropping doesn't update indexes
strategy_data = full_data[full_data.index % 2 == 1].T
weight_data = full_data[full_data.index % 2 == 0].T

# Plotting the number of agents in strategy ratio bins over time
list_of_bins = []
bin_width = 0.10
num_bins = 1/bin_width
for row in strategy_data.iterrows():
    index, data = row
    #bins = [[i * bin_width, (i + 1) * bin_width, 0] for i in range(int(num_bins))]
    bins = [0 for i in range(int(num_bins))]
    for data_point in list(data):
        bin_number = float(data_point) // bin_width
        #bins[int(bin_number)][2] += 1
        bins[int(bin_number)] += 1
    list_of_bins.append(bins)
by_bin = list(zip(*list_of_bins))
plt.figure()
plt.title('Evolution of Agent Strategies')
plt.xlabel('Saved Time Step')
plt.ylabel('Number of Agents in Bin')
for index, bin in enumerate(by_bin):
    label = (round(index*bin_width, 1), round((index+1)*bin_width, 1))
    plt.plot(range(len(bin)), bin, label=label)
plt.legend(loc='upper left')
plt.show()


# Proportion of Interaction Plot over time
proportion_data = proportion_data.values.tolist()[0]
for index, item in enumerate(proportion_data):
    proportion_data[index] = ast.literal_eval(item)
proportion_by_interaction = list(zip(*proportion_data))
plt.figure()
plt.title('Proportion of Interactions')
plt.xlabel('Saved Time Step')
plt.ylabel('Proportion of Interactions')
labels = ['DD', 'DC', 'CD', 'CC']
for index, interaction in enumerate(proportion_by_interaction):
    plt.plot(range(len(interaction)), interaction, label=labels[index])
plt.legend(loc='upper left')
plt.show()


# Plotting link weight ratio bins over time
list_of_bins = []
bin_width = 0.25
num_bins = 1/bin_width
for row in weight_data.iterrows():
    index, data = row
    bins = [0 for i in range(int(num_bins))]
    for item in data:
        item = ast.literal_eval(item)
        norm_item = [float(i) / sum(item) for i in item]  # Should switch to normalizing before save
        for data_point in norm_item:
            bin_number = float(data_point) // bin_width
            bins[int(bin_number)] += 1
        list_of_bins.append(bins)
by_bin = list(zip(*list_of_bins))
plt.figure()
plt.title('Evolution of Weight Ratios')
plt.xlabel('Saved Time Step')
plt.ylabel('Number of Link Weights in Bin')
for index, bin in enumerate(by_bin):
    label = (round(index*bin_width, 2), round((index+1)*bin_width, 2))
    plt.plot(range(len(bin)), bin, label=label)
plt.legend(loc='upper left')
plt.show()


