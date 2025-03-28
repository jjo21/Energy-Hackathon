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

print(tavp.head())

paf_min_len = min(len(paf), len(tavp))
tdap_min_len = min(len(tdap), len(tavp))
for j in range(1, 7):
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
    for i in range(6):
        if i == 0:
            aggtype = 'Mean'
            dfrtype2 = 'DC-H'
        elif i == 1:
            aggtype = 'Max'
            dfrtype2 = 'DC-L'
        elif i == 2:
            aggtype = 'Change in'
            dfrtype2 = 'DR-H'
        elif i == 3:
            aggtype = 'Std Dev of'
            dfrtype2 = 'DR-L'
        elif i == 4:
            aggtype = 'Range of'
            dfrtype2 = 'DM-H'
        elif i == 5:
            aggtype = 'Max Change in'
            dfrtype2 = 'DM-L'
        
        plt.figure()
        plt.scatter(aggregate(paf, 'National Demand Forecast (NDF) - GB (MW)', i)['National Demand Forecast (NDF) - GB (MW)'][:paf_min_len], tavp.iloc[:paf_min_len, j])
        plt.xlabel(f'{aggtype} National Demand Forecast (NDF) - GB (MW)')
        plt.ylabel(f'Volume Requirements Forecast - {dfrtype} - GB (MW)')
        plt.title(f'{aggtype} National Demand Forecast vs {dfrtype} Volume Requirements Forecast')
        plt.savefig(f'plots_kent/{aggtype} National Demand Forecast vs {dfrtype} Volume Requirements Forecast.png')

        plt.figure()
        plt.scatter(aggregate(paf, 'National Demand Forecast (NDF) - GB (MW)', i)['National Demand Forecast (NDF) - GB (MW)'][:paf_min_len], tavp.iloc[:paf_min_len, j + 13])
        plt.xlabel(f'{aggtype} National Demand Forecast (NDF) - GB (MW)')
        plt.ylabel(f'Ancillary Price - {dfrtype} - GB (£/MW/h)')
        plt.title(f'{aggtype} National Demand Forecast vs {dfrtype} Ancillary Price')
        plt.savefig(f'plots_kent/{aggtype} National Demand Forecast vs {dfrtype} Ancillary Price.png')

        plt.figure()
        plt.scatter(tavp.iloc[:, j], tavp.iloc[:, i + 13])
        plt.xlabel(f'Volume Requirements Forecast - {dfrtype} - GB (MW)')
        plt.ylabel(f'Ancillary Price - {dfrtype2} - GB (£/MW/h)')
        plt.title(f'{dfrtype} Volume Requirements Forecast vs {dfrtype2} Ancillary Price')
        plt.savefig(f'plots_kent/{dfrtype} Volume Requirements Forecast vs {dfrtype2} Ancillary Price.png')

        plt.figure()
        plt.scatter(aggregate(tdap, 'Day Ahead Price (EPEX, local) - GB (£/MWh)', i)['Day Ahead Price (EPEX, local) - GB (£/MWh)'][:tdap_min_len], tavp.iloc[:tdap_min_len, j + 12])
        plt.xlabel(f'{aggtype} Day Ahead Price (EPEX, local) - GB (£/MWh)')
        plt.ylabel(f'Ancillary Price - {dfrtype} - GB (£/MW/h)')
        plt.title(f'EPEX {aggtype} Day Ahead Price vs {dfrtype} Ancillary Price')
        plt.savefig(f'plots_kent/EPEX {aggtype} Day Ahead Price vs {dfrtype} Ancillary Price.png')

        plt.figure()
        plt.scatter(aggregate(tdap, 'Day Ahead Price (N2EX, local) - GB (£/MWh)', i)['Day Ahead Price (N2EX, local) - GB (£/MWh)'][:tdap_min_len], tavp.iloc[:tdap_min_len, j + 12])
        plt.xlabel(f'{aggtype} Day Ahead Price (N2EX, local) - GB (£/MWh)')
        plt.ylabel(f'Ancillary Price - {dfrtype} - GB (£/MW/h)')
        plt.title(f'N2EX {aggtype} Day Ahead Price vs {dfrtype} Ancillary Price')
        plt.savefig(f'plots_kent/N2EX {aggtype} Day Ahead Price vs {dfrtype} Ancillary Price.png')

        plt.figure()
        plt.scatter(aggregate(tdap, 'Day Ahead Price (EPEX, local) - GB (£/MWh)', i)['Day Ahead Price (EPEX, local) - GB (£/MWh)'][:tdap_min_len], tavp.iloc[:tdap_min_len, j + 12])
        plt.xlabel(f'{aggtype} Day Ahead Price (EPEX, local) - GB (£/MWh)')
        plt.ylabel(f'Volume Requirements Forecast - {dfrtype} - GB (MW)')
        plt.title(f'EPEX {aggtype} Day Ahead Price vs {dfrtype} Volume Requirements Forecast')
        plt.savefig(f'plots_kent/EPEX {aggtype} Day Ahead Price vs {dfrtype} Volume Requirements Forecast.png')

        plt.figure()
        plt.scatter(aggregate(tdap, 'Day Ahead Price (N2EX, local) - GB (£/MWh)', i)['Day Ahead Price (N2EX, local) - GB (£/MWh)'][:tdap_min_len], tavp.iloc[:tdap_min_len, j + 12])
        plt.xlabel(f'{aggtype} Day Ahead Price (N2EX, local) - GB (£/MWh)')
        plt.ylabel(f'Volume Requirements Forecast - {dfrtype} - GB (MW)')
        plt.title(f'N2EX {aggtype} Day Ahead Price vs {dfrtype} Volume Requirements Forecast')
        plt.savefig(f'plots_kent/N2EX {aggtype} Day Ahead Price vs {dfrtype} Volume Requirements Forecast.png')

