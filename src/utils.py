import pandas as pd
import matplotlib.pyplot as plt

def plot(csv_file, out_file, title):
    df = pd.read_csv(csv_file)
    plt.plot(df["loss"])
    plt.title(title)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.savefig(out_file)
    plt.close()
