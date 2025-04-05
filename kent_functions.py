import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_excel_or_date(x):
    try:
        # Try to convert if it's a float or int and looks like a serial date
        if isinstance(x, (str, float)) and 10000 < float(x) < 60000:
            return pd.Timestamp('1899-12-30') + pd.to_timedelta(float(x), unit='D')
        # Otherwise treat it as a normal datetime string
        return pd.to_datetime(x, dayfirst=True)
    except:
        return pd.NaT

def corr_grid(df1, df2, col1, col2, label1, label2, header=None, figsize=(18, 18), filename=None):
    n = len(col2)
    m = len(col1)
    fig, axs = plt.subplots(n, m, figsize=figsize, squeeze=False)

    for i in range(n):
        for j in range(m):
            ax = axs[i, j]
            x = df1[col1[j]]
            y = df2[col2[i]]
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]
            corr = x.corr(y)
            r2 = corr ** 2

            # Scatter plot
            ax.scatter(x, y, s=10, alpha=0.5)

            # Title with correlation
            ax.set_title(f'r = {corr:.2f}, r^2 = {r2:.2f}', fontsize=13, pad=1)

            # Border color based on absolute correlation
            abs_r2 = abs(r2)
            if abs_r2 > 0.7:
                color = 'red'
            elif abs_r2 > 0.3:
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
    min_ = groups.min()
    change = groups.agg(lambda x: x.iloc[-1] - x.iloc[0] if len(x) >= 2 else np.nan)
    range_ = groups.agg(lambda x: x.max() - x.min() if len(x) >= 2 else np.nan)
    max_change = groups.agg(lambda x: np.max(np.abs(np.diff(x))) if len(x) >= 2 else np.nan)

    # Rename columns
    mean.columns = ['Mean']
    max_.columns = ['Max']
    min_.columns = ['Min']
    change.columns = ['Change']
    range_.columns = ['Range']
    max_change.columns = ['Max Change']

    # Join all into one DataFrame
    df_agg = pd.concat([mean, max_, min_, change, range_, max_change], axis=1).reset_index()
    
    return df_agg

def aggregate(df, value_col, method):
    df = df.copy()
    df.set_index('GMT Time', inplace=True)
    groups = df[[value_col]].resample('4h', label='right', closed="right", origin="epoch", offset='3h')

    # Compute each aggregate
    if method == 'mean':
        ndf = groups.mean()
        ndf.columns = ['Mean']
    elif method == 'max':
        ndf = groups.max()
        ndf.columns = ['Max']
    elif method == 'min':
        ndf = groups.min()
        ndf.columns = ['Min']
    
    return ndf

def get_efa_block(dt):
    hour = dt.hour
    if hour >= 23 or hour < 3:
        return 1
    elif 3 <= hour < 7:
        return 2
    elif 7 <= hour < 11:
        return 3
    elif 11 <= hour < 15:
        return 4
    elif 15 <= hour < 19:
        return 5
    elif 19 <= hour < 23:
        return 6

def lag_plot_matrix(series, freq, max_lag, label, header, figsize=(14, 10), filename=None):
    if freq == 'EFA':
        jump = 1
    elif freq == 'D':
        jump = 6
    elif freq == 'W':
        jump = 42
    elif freq == 'Q':
        jump = 546
    
    data_cols = series.columns[1:]  # skip timestamp
    nrows = len(data_cols)
    ncols = max_lag
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    for i, col in enumerate(data_cols):          # i: row index, col: column name
        for j, lag in enumerate(range(jump, (max_lag + 1) * jump, jump)):  # j: col index
            lagged = series[col].shift(lag)
            aligned = pd.concat([lagged, series[col]], axis=1).dropna()
            x, y = aligned.iloc[:, 0], aligned.iloc[:, 1]
            r = x.corr(y)
            r2 = r ** 2

            ax = axs[i, j]
            ax.scatter(x, y, alpha=0.5, s=10)

            color = 'red' if abs(r2) > 0.7 else 'orange' if abs(r2) > 0.3 else 'green'
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

            ax.set_title(f'r = {r:.2f}, r^2 = {r2:.2f}', fontsize=13, pad=1)
            #ax.set_title(f'Lag {lag} (r = {r:.2f})', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    
            if i == nrows - 1:
                ax.set_xlabel(f'Lag {j + 1}', fontsize=10)
            if j == 0:
                ax.set_ylabel(label[i], fontsize=10)

    fig.suptitle(header, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if filename:
        plt.savefig(filename)
    plt.close()

def time_filters(df):
    df['EFA'] = df['GMT Time'].apply(get_efa_block)
    df['Quarter'] = df['GMT Time'].dt.quarter
    df['Day'] = df['GMT Time'].dt.dayofweek
    df['Weekend'] = np.where(df['Day'] < 5, 0, 1)
    df['Post EAC'] = np.where(df['GMT Time'] < pd.Timestamp('2023-11-03'), 0, 1)
    return df