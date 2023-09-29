import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os

dir = "D:/Università/Tesi/Distributed Learning/Metrics"

def plot():
    files = os.listdir(dir)
    num_workers = len(files)
    plt.figure(figsize=(10, 10))
    for i in range(len(files)):
        filepath = os.path.join(dir, files[i])
        
        with open(filepath, "r") as file:
            data = json.loads(file.read())
            worker_id = data["workerID"]
            metrics = data["metrics"]
        
        if worker_id == 0:
            title = "Chief"
        else:
            title = f"Worker{worker_id}"
            
        labels = ['Loss', 'Accuracy', 'Train_MSE', 'Train_MAE', 'Val_MSE', 'Val_MAE']
        df = pd.DataFrame(metrics)
        df.plot(kind='bar', title='Metrics', grid=True)
        plt.subplot(num_workers, 1, i+1, title = title)
        plt.xticks(ticks = np.arange(6), rotation = 0, labels = labels)
    plt.show()

def create_subplots_from_files(directory = dir):
    files = os.listdir(directory)
    num_files = len(files)
    num_cols = num_files

    fig, axes = plt.subplots(1, num_cols, figsize=(15, 5))
    fig.tight_layout(pad=3.0)

    colors = plt.cm.get_cmap('tab10').colors

    for i, file_name in enumerate(files):
        file_path = os.path.join(directory, file_name)
        ax = axes[i] if num_files > 1 else axes

        with open(file_path, 'r') as file:
            data = json.load(file)

        data = data["metrics"]
        metrics = list(data.keys())
        
        labels = ['Loss', 'Accuracy', 'MSE', 'MAE', 'Val_Loss', 'Val_Acc', 'Val_MSE', 'Val_MAE']
        last_values = [data[metric][-1] for metric in metrics]

        df = pd.DataFrame({'Metric': labels, 'Last Value': last_values})

        color = colors[i % len(colors)]

        df.plot(x='Metric', y='Last Value', kind='bar', ax=ax, color=color, legend=False)

        if file_name[:-5] == "0":
            title = "Chief"
        else:
            title = f"Worker {file_name[:-5]}"
        ax.set_title(title)
        ax.grid(True)
        ax.set_xticklabels(df['Metric'], rotation=45)
        ax.axhline(1, color='red', linestyle='--')

    plt.show()


if __name__ == '__main__':
    create_subplots_from_files()
    #plot()