import pandas as pd
import matplotlib.pyplot as plt
import sys

# only used for visualization in thesis

file_path1 = "/home/andi/Desktop/LRZ Sync+Share/Master/5. Semester/Master thesis/Thesis/KLvsMSE/with_noise.csv"  # Replace with your first file path
file_path2 = "/home/andi/Desktop/LRZ Sync+Share/Master/5. Semester/Master thesis/Thesis/KLvsMSE/without_noise.csv"  # Replace with your second file path
file_path1 = "/home/len1218/documents/BT/framework/logs/experiment_test/episode_reward.csv"
file_path2 = "/home/len1218/documents/BT/framework/logs/experiment_test/episode_reward.csv"
file_path1 = file_path2 = sys.argv[1]

data1 = pd.read_csv(file_path1)  # First dataset
data2 = pd.read_csv(file_path2)  # Second dataset

# Extract Step and Value for both datasets
steps1, values1 = data1['Step'], data1['Value']
steps2, values2 = data2['Step'], data2['Value']

# Create the plot
plt.figure(figsize=(10, 6))

# Add the first dataset
#plt.plot(steps1, values1, c='blue', label='Collective reward')
plt.plot(steps1, values1, c='blue', label='KL')

# Add the second dataset
#plt.plot(steps2, values2, c='red', label='Standard reward')
plt.plot(steps2, values2, c='red', label='MSE')

# Customize the plot
plt.title('Multiple Graphs in One Plot', fontsize=14)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend()
plt.grid(True)

#plt.savefig(f'/home/andi/Desktop/mtrl/additional_scripts/graphs/KLvsMSE.svg', format='svg')
plt.savefig(f'graphs/graph.svg', format='svg')

# Show the plot
plt.show()
