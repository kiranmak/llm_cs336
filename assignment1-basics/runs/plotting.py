import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the root folder holding all your learning rate subfolders
log_dir = './logs'

# List all metrics.jsonl files under log_dir
jsonl_files = glob.glob(os.path.join(log_dir, '*/metrics.jsonl'))

data = []
for file_path in jsonl_files:
    # Extract directory name as the experiment identifier
    dir_name = os.path.basename(os.path.dirname(file_path))
    
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            # Add the directory name to the record
            record['dir_name'] = dir_name
            data.append(record)

df = pd.DataFrame(data)

# Filter for the specific metric (e.g., 'train/loss_avg' or 'val/loss')
metric_name = 'train/loss_avg'
if metric_name in df.columns:
    df_filtered = df[['step', 'dir_name', metric_name]].dropna()
    df_filtered = df_filtered.rename(columns={metric_name: 'value'})

    # Plot all learning rates on a single graph
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x='step', y='value', hue='dir_name', linewidth=2)

    plt.title(f'Comparison of {metric_name} Across Learning Rates')
    plt.xlabel('Training Steps')
    plt.ylabel('Value')
    plt.legend(title='Experiment Folder', loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('learning_rate_comparison.png', dpi=300)
    print("Saved plot to learning_rate_comparison.png")
else:
    print(f"Metric {metric_name} not found in the logged metrics.")
