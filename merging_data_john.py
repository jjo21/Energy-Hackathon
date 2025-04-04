
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_exploration_funcs as ef

#%%

avp = pd.read_csv('2023\Ancillary Volumes & Prices (4H).csv')
dap = pd.read_csv('2023\Day-Ahead Price (1H).csv').dropna()
df = pd.read_csv('2023\Prices & Forecasts (HH).csv')

# %%

avp['GMT Time'] = pd.to_datetime(avp['GMT Time'])
avp['Forecast Time (DFR Volume Requirement)'] = (avp['GMT Time'] - pd.Timedelta(days=1)).dt.normalize()
avp['Forecast Time (Prices / Volumes Accepted)'] = avp['Forecast Time (DFR Volume Requirement)'] + pd.Timedelta(hours=14, minutes=30)

# %%

# merging the dataframe on itself to shift the lagged variables

avp2 = avp.copy()

avp2['Shifted Time'] = avp2['GMT Time'] + pd.Timedelta(days=1)
avp_new = pd.merge(avp, avp2, left_on = 'GMT Time', right_on = 'Shifted Time')

# %%
