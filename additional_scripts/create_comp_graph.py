import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# CSV paths from command-line arguments
file_paths = sys.argv[1:]
if not file_paths:
    print("Usage: python plot_multi.py file1.csv file2.csv ...")
    sys.exit(1)

# Output folder
output_dir = "graphs/combined"
os.makedirs(output_dir, exist_ok=True)

# Load all CSVs
dfs = []
names = []
for path in file_paths:
    df = pd.read_csv(path)
    dfs.append(df)
    names.append(os.path.splitext(os.path.basename(path))[0])

# Collect all reward columns (skip 'Step')
all_columns = sorted(
    set(col for df in dfs for col in df.columns if col != 'Step')
)

steps_list = [df['Step'] for df in dfs]

# Plot each reward column
for col in all_columns:
    plt.figure(figsize=(10, 6))

    for df, steps, name in zip(dfs, steps_list, names):
        if col in df.columns:
            plt.plot(steps, df[col], label=name)

    plt.title(f'Comparison of {col}', fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.legend()
    plt.grid(True)

    out_file = os.path.join(output_dir, f'{col}.svg')
    plt.savefig(out_file, format='svg')
    plt.close()

print(f"Combined graphs saved to {output_dir}/")

