import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def corr_grid(df1, df2, col1, col2, label1=None, label2=None, header=None, figsize=(18, 18), filename=None):
    n = len(col1)
    fig, axs = plt.subplots(n, n, figsize=figsize)

    # Optional labels
    if label1 is None:
        label1 = col1
    if label2 is None:
        label2 = col2

    for i in range(n):
        for j in range(n):
            ax = axs[i, j]
            x = df1[col1[j]]
            y = df2[col2[i]]
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]
            corr = x.corr(y)

            # Scatter plot
            ax.scatter(x, y, s=10, alpha=0.5)

            # Title with correlation
            ax.set_title(f'r = {corr:.2f}', fontsize=13, pad=1)

            # Border color based on absolute correlation
            abs_corr = abs(corr)
            if abs_corr > 0.7:
                color = 'red'
            elif abs_corr > 0.3:
                color = 'orange'
            else:
                color = 'green'

            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

            ax.set_xticks([])
            ax.set_yticks([])

            if i == n - 1:
                ax.set_xlabel(label1[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(label2[i], fontsize=10)

    plt.suptitle(header, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if filename:
        plt.savefig(filename)
    plt.close()

def aggregate_all(df, value_col):
    df = df.copy()
    df.set_index('GMT Time', inplace=True)
    groups = df[[value_col]].resample('4h', label='right', closed="right", origin="epoch", offset='3h')

    # Compute each aggregate
    mean = groups.mean()
    max_ = groups.max()
    change = groups.agg(lambda x: x.iloc[-1] - x.iloc[0] if len(x) >= 2 else np.nan)
    std = groups.std()
    range_ = groups.agg(lambda x: x.max() - x.min() if len(x) >= 2 else np.nan)
    max_change = groups.agg(lambda x: np.max(np.abs(np.diff(x))) if len(x) >= 2 else np.nan)

    # Rename columns
    mean.columns = ['Mean']
    max_.columns = ['Max']
    change.columns = ['Change']
    std.columns = ['Std Dev']
    range_.columns = ['Range']
    max_change.columns = ['Max Change']

    # Join all into one DataFrame
    df_agg = pd.concat([mean, max_, change, std, range_, max_change], axis=1).reset_index()
    
    return df_agg