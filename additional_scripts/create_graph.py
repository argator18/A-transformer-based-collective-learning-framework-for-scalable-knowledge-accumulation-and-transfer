import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# CSV path from command-line argument
file_path = sys.argv[1]
data = pd.read_csv(file_path)

# Get base name of the input file without extension
print(file_path)
input_base = os.path.splitext(os.path.basename(file_path))[0]

# Ensure an output folder for the graphs
output_dir = f"graphs/{input_base}"
os.makedirs(output_dir, exist_ok=True)

# Extract steps
steps = data['Step']

# Iterate over each reward column (skip 'Step')
for col in data.columns[1:]:
    values = data[col]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, c='blue', label=col)

    plt.title(f'{input_base} | {col}', fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.legend()
    plt.grid(True)

    # Save each graph as an SVG
    out_file = os.path.join(output_dir, f'{col}.svg')
    plt.savefig(out_file, format='svg')
    plt.close()  # Close to avoid overlapping plots

print(f"Graphs saved to {output_dir}/")
