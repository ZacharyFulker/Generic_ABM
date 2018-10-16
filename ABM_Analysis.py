import pandas as pd

full_data = pd.read_csv('run_results.csv', header=None)
strategy_data = full_data[full_data.index % 2 == 0].T
weight_data = full_data[full_data.index % 2 == 1].T

# Plotting the number of agents in strategy ratio bins over time
bin_width = 0.10
num_bins = 1/bin_width
for row in strategy_data.iterrows():
    index, data = row
    #bins = [[i * bin_width, (i + 1) * bin_width, 0] for i in range(int(num_bins))]
    bins = [0 for i in range(int(num_bins))]
    for data_point in list(data):
        bin_number = float(data_point) // bin_width
        print(bin_number)
        #bins[int(bin_number)][2] += 1
        bins[int(bin_number)] += 1
    exit()