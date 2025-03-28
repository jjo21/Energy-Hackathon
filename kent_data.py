#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

avp = pd.read_csv('2023\Ancillary Volumes & Prices (4H).csv')
dap = pd.read_csv('2023\Day-Ahead Price (1H).csv').dropna()
df1 = pd.read_csv('2023\Prices & Forecasts (HH).csv')

avp2 = pd.read_csv('2024\Ancillary Volumes & Prices (4H).csv')
dap2 = pd.read_csv('2024\Day-Ahead Price (1H).csv')
df2 = pd.read_csv('2024\Prices & Forecasts (HH).csv')

tavp = pd.concat([avp, avp2], ignore_index=True)
tavp['GMT Time'] = pd.to_datetime(tavp['GMT Time'], dayfirst=True)

tdap = pd.concat([dap, dap2], ignore_index=True)
tdap['GMT Time'] = pd.to_datetime(tdap['GMT Time'], dayfirst=True)

paf = pd.concat([df1, df2], ignore_index=True)
paf['GMT Time'] = pd.to_datetime(paf['GMT Time'], dayfirst=True)
paf['Hour'] = paf['GMT Time'].dt.hour
paf['Day'] = paf['GMT Time'].dt.month

#0 = mean, 1 = max, 2 = change, 3 = std, 4 = range, 5 = maxchange
def aggregate(df, value_col, agg_type=0):
    df = df.copy()
    df.set_index('GMT Time', inplace=True)
    groups = df[[value_col]].resample('4h', label='right', closed="right", origin="epoch", offset='3h')
    if agg_type == 0:
        df_agg = groups.mean()
    elif agg_type == 1:
        df_agg = groups.max()
    elif agg_type == 2:
        df_agg = groups.agg(lambda x: x.iloc[-1] - x.iloc[0])
    elif agg_type == 3:
        df_agg = groups.std()
    elif agg_type == 4:
        df_agg = groups.agg(lambda x: x.max() - x.min())
    elif agg_type == 5:
        df_agg = groups.agg(lambda x: np.max(np.abs(np.diff(x))) if len(x) >= 2 else np.nan)
    else:
        raise ValueError("Unsupported aggregation type")
    return df_agg.reset_index()

#print(aggregate(tdap, 'Day Ahead Price (EPEX, local) - GB (£/MWh)', 'mean'))

paf_min_len = min(len(paf), len(tavp))
for j in range(1, tavp.shape[1]):
    for i in range(6):
        plt.figure()
        plt.scatter(aggregate(paf, 'National Demand Forecast (NDF) - GB (MW)', i)['National Demand Forecast (NDF) - GB (MW)'][:paf_min_len], tavp.iloc[:paf_min_len, j])
        if j == 1:
            dfrtype = 'DC-H'
        elif j == 2:
            dfrtype = 'DC-L'
        elif j == 3:
            dfrtype = 'DR-H'
        elif j == 4:
            dfrtype = 'DR-L'
        elif j == 5:
            dfrtype = 'DM-H'
        elif j == 6:
            dfrtype = 'DM-L'
        if i == 0:
            aggtype = 'Mean'
        elif i == 1:
            aggtype = 'Max'
        elif i == 2:
            aggtype = 'Change in'
        elif i == 3:
            aggtype = 'Std Dev of'
        elif i == 4:
            aggtype = 'Range of'
        elif i == 5:
            aggtype = 'Max Change in'
        plt.xlabel(f'{aggtype} National Demand Forecast (NDF) - GB (MW)')
        plt.ylabel(f'Volume Requirements Forecast - {dfrtype} - GB (MW)')
        plt.title(f'{aggtype} National Demand Forecast vs {dfrtype} Volume Requirements Forecast')
        plt.savefig(f'plots_kent/{aggtype} National Demand Forecast vs {dfrtype} Volume Requirements Forecast.png')

plt.figure()
plt.scatter(tavp['Volume Requirements Forecast - DC-H - GB (MW)'], tavp['Ancillary Price - DC-H - GB (£/MW/h)'])
plt.xlabel('Volume Requirements Forecast - DC-H - GB (MW)')
plt.ylabel('Ancillary Price - DC-H - GB (£/MW/h)')
plt.title('Volume Requirements Forecast DC-H vs Ancillary Price')
plt.savefig('plots_kent/Volume Requirements Forecast DC-H vs Ancillary Price.png')

tdap_min_len = min(len(tdap), len(tavp))
plt.figure()
plt.scatter(aggregate(tdap, 'Day Ahead Price (EPEX, local) - GB (£/MWh)', 'mean')['Day Ahead Price (EPEX, local) - GB (£/MWh)'][:tdap_min_len], tavp['Ancillary Price - DC-H - GB (£/MW/h)'][:tdap_min_len])
plt.xlabel('Day Ahead Price (EPEX, local) - GB (£/MWh)')
plt.ylabel('Ancillary Price - DC-H - GB (£/MW/h)')
plt.title('Day Ahead Price vs Ancillary Price')
plt.savefig('plots_kent/Day Ahead Price vs Ancillary Price.png')

plt.figure()
plt.scatter(aggregate(tdap, 'Day Ahead Price (EPEX, local) - GB (£/MWh)', 'mean')['Day Ahead Price (EPEX, local) - GB (£/MWh)'][:tdap_min_len], tavp['Volume Requirements Forecast - DC-H - GB (MW)'][:tdap_min_len])
plt.xlabel('Day Ahead Price (EPEX, local) - GB (£/MWh)')
plt.ylabel('Volume Requirements Forecast - DC-H - GB (MW)')
plt.title('Day Ahead Price vs Volume Requirements Forecast DC-H')
plt.savefig('plots_kent/Day Ahead Price vs Volume Requirements Forecast DC-H.png')

plt.show()
