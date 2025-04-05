
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_exploration_funcs as ef

#%%

avp = pd.read_csv('2023\Ancillary Volumes & Prices (4H).csv')
# avp2 = pd.read_csv('2024\Ancillary Volumes & Prices (4H).csv')
dap_orig = pd.read_csv('2023\Day-Ahead Price (1H).csv').dropna()
df = pd.read_csv('2023\Prices & Forecasts (HH).csv')
# df2 = pd.read_csv('2024\Prices & Forecasts (HH).csv')
# avp = pd.concat([avp, avp2], ignore_index=True)
# df = pd.concat([df, df2], ignore_index=True)

# %%

avp['GMT Time'] = pd.to_datetime(avp['GMT Time'], dayfirst=True)
avp['Forecast Time (DFR Volume Requirement)'] = (avp['GMT Time'] - pd.Timedelta(days=1)).dt.normalize()
avp['Forecast Time (Prices / Volumes Accepted)'] = avp['Forecast Time (DFR Volume Requirement)'] + pd.Timedelta(hours=14, minutes=30)

#%%

DC = avp[['GMT Time', 'Forecast Time (DFR Volume Requirement)', 'Volume Requirements Forecast - DC-H - GB (MW)', 'Volume Requirements Forecast - DC-L - GB (MW)', 'Forecast Time (Prices / Volumes Accepted)', 'Ancillary Volume Accepted - DC-H - GB (MW)', 'Ancillary Volume Accepted - DC-L - GB (MW)', 'Ancillary Price - DC-H - GB (£/MW/h)', 'Ancillary Price - DC-L - GB (£/MW/h)']]
DM = avp[['GMT Time', 'Forecast Time (DFR Volume Requirement)', 'Volume Requirements Forecast - DM-H - GB (MW)', 'Volume Requirements Forecast - DM-L - GB (MW)', 'Forecast Time (Prices / Volumes Accepted)', 'Ancillary Volume Accepted - DM-H - GB (MW)', 'Ancillary Volume Accepted - DM-L - GB (MW)', 'Ancillary Price - DM-H - GB (£/MW/h)', 'Ancillary Price - DM-L - GB (£/MW/h)']]
DR = avp[['GMT Time', 'Forecast Time (DFR Volume Requirement)', 'Volume Requirements Forecast - DR-H - GB (MW)', 'Volume Requirements Forecast - DR-L - GB (MW)', 'Forecast Time (Prices / Volumes Accepted)', 'Ancillary Volume Accepted - DR-H - GB (MW)', 'Ancillary Volume Accepted - DR-L - GB (MW)', 'Ancillary Price - DR-H - GB (£/MW/h)', 'Ancillary Price - DR-L - GB (£/MW/h)']]

#%%

# plot the prices against the volume requirement 

plt.scatter(DC['Volume Requirements Forecast - DC-H - GB (MW)'], DC['Ancillary Price - DC-H - GB (£/MW/h)'])
plt.xlabel('Volume Requirements Forecast - DC-H - GB (MW)')
plt.ylabel('Ancillary Price - DC-H - GB (£/MW/h)')
plt.title('Volume Requirements Forecast vs Ancillary Price for DC-H')
plt.grid()
plt.tight_layout()
plt.show()

#%%

plt.scatter(DC['Volume Requirements Forecast - DC-L - GB (MW)'], DC['Ancillary Price - DC-L - GB (£/MW/h)'])
plt.xlabel('Volume Requirements Forecast - DC-L - GB (MW)')
plt.ylabel('Ancillary Price - DC-L - GB (£/MW/h)')
plt.title('Volume Requirements Forecast vs Ancillary Price for DC-L')
plt.grid()
plt.tight_layout()
plt.show()

#%%

plt.scatter(DR['Volume Requirements Forecast - DR-H - GB (MW)'], DR['Ancillary Price - DR-H - GB (£/MW/h)'])
plt.xlabel('Volume Requirements Forecast - DR-H - GB (MW)')
plt.ylabel('Ancillary Price - DR-H - GB (£/MW/h)')
plt.title('Volume Requirements Forecast vs Ancillary Price for DR-H')
plt.grid()
plt.tight_layout()
plt.show()

#%%

plt.scatter(DR['Volume Requirements Forecast - DR-L - GB (MW)'], DR['Ancillary Price - DR-L - GB (£/MW/h)'])
plt.xlabel('Volume Requirements Forecast - DR-L - GB (MW)')
plt.ylabel('Ancillary Price - DR-L - GB (£/MW/h)')
plt.title('Volume Requirements Forecast vs Ancillary Price for DR-L')
plt.grid()
plt.tight_layout()
plt.show()

#%%

plt.scatter(DM['Volume Requirements Forecast - DM-H - GB (MW)'], DM['Ancillary Price - DM-H - GB (£/MW/h)'])
plt.xlabel('Volume Requirements Forecast - DM-H - GB (MW)')
plt.ylabel('Ancillary Price - DM-H - GB (£/MW/h)')
plt.title('Volume Requirements Forecast vs Ancillary Price for DM-H')
plt.grid()
plt.tight_layout()
plt.show()

#%%

plt.scatter(DM['Volume Requirements Forecast - DM-H - GB (MW)'], DM['Ancillary Price - DM-H - GB (£/MW/h)'])
plt.xlabel('Volume Requirements Forecast - DM-H - GB (MW)')
plt.ylabel('Ancillary Price - DM-H - GB (£/MW/h)')
plt.title('Volume Requirements Forecast vs Ancillary Price for DM-H')
plt.grid()
plt.tight_layout()
plt.show()

#%%

# plot the difference between volume required and volume accepted 
plt.plot(DC['Volume Requirements Forecast - DC-H - GB (MW)'] - DC['Ancillary Volume Accepted - DC-H - GB (MW)'])
plt.show()
plt.scatter(DC['Volume Requirements Forecast - DC-H - GB (MW)'] - DC['Ancillary Volume Accepted - DC-H - GB (MW)'], DC['Ancillary Price - DC-H - GB (£/MW/h)'])
plt.xlabel('Volume Requirements Forecast - DC-H - GB (MW) - Ancillary Volume Accepted - DC-H - GB (MW)')
plt.ylabel('Ancillary Price - DC-H - GB (£/MW/h)')
plt.title('Volume Requirements Forecast vs Ancillary Price for DC-H')
plt.grid()
plt.tight_layout()
plt.show()

#%%

# plot the difference between volume required and volume accepted 
plt.plot(DC['Volume Requirements Forecast - DC-L - GB (MW)'] - DC['Ancillary Volume Accepted - DC-L - GB (MW)'])
plt.show()
plt.scatter(DC['Volume Requirements Forecast - DC-L - GB (MW)'] - DC['Ancillary Volume Accepted - DC-L - GB (MW)'], DC['Ancillary Price - DC-L - GB (£/MW/h)'])
plt.xlabel('Volume Requirements Forecast - DC-L - GB (MW) - Ancillary Volume Accepted - DC-L - GB (MW)')
plt.ylabel('Ancillary Price - DC-L - GB (£/MW/h)')
plt.title('Volume Requirements Forecast vs Ancillary Price for DC-L')
plt.grid()
plt.tight_layout()
plt.show()

#%%

# plot the difference between volume required and volume accepted 
plt.plot(DR['Volume Requirements Forecast - DR-H - GB (MW)'] - DR['Ancillary Volume Accepted - DR-H - GB (MW)'])
plt.show()
plt.scatter(DR['Volume Requirements Forecast - DR-H - GB (MW)'] - DR['Ancillary Volume Accepted - DR-H - GB (MW)'], DR['Ancillary Price - DR-H - GB (£/MW/h)'])
plt.xlabel('Volume Requirements Forecast - DR-H - GB (MW) - Ancillary Volume Accepted - DR-H - GB (MW)')
plt.ylabel('Ancillary Price - DR-H - GB (£/MW/h)')
plt.title('Volume Requirements Forecast vs Ancillary Price for DR-H')
plt.grid()
plt.tight_layout()
plt.show()

#%%

# plot the difference between volume required and volume accepted 
plt.plot(DR['Volume Requirements Forecast - DR-L - GB (MW)'] - DR['Ancillary Volume Accepted - DR-L - GB (MW)'])
plt.show()
plt.scatter(DR['Volume Requirements Forecast - DR-L - GB (MW)'] - DR['Ancillary Volume Accepted - DR-L - GB (MW)'], DR['Ancillary Price - DR-L - GB (£/MW/h)'])
plt.xlabel('Volume Requirements Forecast - DR-L - GB (MW) - Ancillary Volume Accepted - DR-L - GB (MW)')
plt.ylabel('Ancillary Price - DR-L - GB (£/MW/h)')
plt.title('Volume Requirements Forecast vs Ancillary Price for DR-L')
plt.grid()
plt.tight_layout()
plt.show()

#%%

# plot the difference between volume required and volume accepted 
plt.plot(DM['Volume Requirements Forecast - DM-H - GB (MW)'] - DM['Ancillary Volume Accepted - DM-H - GB (MW)'])
plt.show()
plt.scatter(DM['Volume Requirements Forecast - DM-H - GB (MW)'] - DM['Ancillary Volume Accepted - DM-H - GB (MW)'], DM['Ancillary Price - DM-H - GB (£/MW/h)'])
plt.xlabel('Volume Requirements Forecast - DM-H - GB (MW) - Ancillary Volume Accepted - DM-H - GB (MW)')
plt.ylabel('Ancillary Price - DM-H - GB (£/MW/h)')
plt.title('Volume Requirements Forecast vs Ancillary Price for DM-H')
plt.grid()
plt.tight_layout()
plt.show()

#%%

# plot the difference between volume required and volume accepted 
plt.plot(DM['Volume Requirements Forecast - DM-L - GB (MW)'] - DM['Ancillary Volume Accepted - DM-L - GB (MW)'])
plt.show()
plt.scatter(DM['Volume Requirements Forecast - DM-L - GB (MW)'] - DM['Ancillary Volume Accepted - DM-L - GB (MW)'], DM['Ancillary Price - DM-L - GB (£/MW/h)'])
plt.xlabel('Volume Requirements Forecast - DM-L - GB (MW) - Ancillary Volume Accepted - DM-L - GB (MW)')
plt.ylabel('Ancillary Price - DM-L - GB (£/MW/h)')
plt.title('Volume Requirements Forecast vs Ancillary Price for DM-L')
plt.grid()
plt.tight_layout()
plt.show()

# %%

# shift the volumes accepted column by 1 and merge it onto the original data frame 
# make nan values 0
DC_shift = DC.copy()
DC_shift['Ancillary Volume Accepted - DC-H - GB (MW)'] = DC_shift['Ancillary Volume Accepted - DC-H - GB (MW)'].shift(1).fillna(0)
DC_shift['Ancillary Volume Accepted - DC-L - GB (MW)'] = DC_shift['Ancillary Volume Accepted - DC-L - GB (MW)'].shift(1).fillna(0)

# %%

plt.scatter(DC_shift['Ancillary Volume Accepted - DC-H - GB (MW)'], DC_shift['Ancillary Price - DC-H - GB (£/MW/h)'])
plt.xlabel('Ancillary Volume Accepted - DC-H - GB (MW)')
plt.ylabel('Ancillary Price - DC-H - GB (£/MW/h)')
plt.title('Ancillary Volume Accepted vs Ancillary Price for DC-H')
plt.grid()
plt.tight_layout()
plt.show()

#%%

plt.scatter(DC_shift['Ancillary Volume Accepted - DC-L - GB (MW)'], DC_shift['Ancillary Price - DC-L - GB (£/MW/h)'])
plt.xlabel('Ancillary Volume Accepted - DC-L - GB (MW)')
plt.ylabel('Ancillary Price - DC-L - GB (£/MW/h)')
plt.title('Ancillary Volume Accepted vs Ancillary Price for DC-L')
plt.grid()
plt.tight_layout()
plt.show()

# %%

DR_shift = DR.copy()
DR_shift['Ancillary Volume Accepted - DR-H - GB (MW)'] = DR_shift['Ancillary Volume Accepted - DR-H - GB (MW)'].shift(1).fillna(0)
DR_shift['Ancillary Volume Accepted - DR-L - GB (MW)'] = DR_shift['Ancillary Volume Accepted - DR-L - GB (MW)'].shift(1).fillna(0)

# %%

plt.scatter(DR_shift['Ancillary Volume Accepted - DC-H - GB (MW)'], DR_shift['Ancillary Price - DC-H - GB (£/MW/h)'])
plt.xlabel('Ancillary Volume Accepted - DC-H - GB (MW)')
plt.ylabel('Ancillary Price - DC-H - GB (£/MW/h)')
plt.title('Ancillary Volume Accepted vs Ancillary Price for DC-H')
plt.grid()
plt.tight_layout()
plt.show()

#%%

plt.scatter(DR_shift['Ancillary Volume Accepted - DC-L - GB (MW)'], DR_shift['Ancillary Price - DC-L - GB (£/MW/h)'])
plt.xlabel('Ancillary Volume Accepted - DC-L - GB (MW)')
plt.ylabel('Ancillary Price - DC-L - GB (£/MW/h)')
plt.title('Ancillary Volume Accepted vs Ancillary Price for DC-L')
plt.grid()
plt.tight_layout()
plt.show()

#%%

'''

Now looking for correlation amongst day ahead price 

1. Aggregating the numbers 
2. Using the last price 

'''

dap = dap_orig.copy()
DC_merge = DC.copy()
dap['GMT Time'] = pd.to_datetime(dap['GMT Time'], dayfirst=True)
dap['Forecast Time'] = (dap['GMT Time'] - pd.Timedelta(days=1)).dt.normalize()
dap['Forecast Time'] = dap['Forecast Time'] + pd.Timedelta(hours=9, minutes=10)
dap['date'] = dap['GMT Time'].apply(ef.shift_efa).dt.date
dap['EFA Block'] = dap['GMT Time'].dt.hour.apply(ef.get_efa_block)

#%%

n2ex = dap.groupby(['date', 'EFA Block'])['Day Ahead Price (N2EX, local) - GB (£/MWh)'].mean().reset_index()
epex = dap.groupby(['date', 'EFA Block'])['Day Ahead Price (EPEX, local) - GB (£/MWh)'].mean().reset_index()
DC_merge['EFA Block'] = DC_merge['GMT Time'].dt.hour.apply(ef.get_efa_block2)
DC_merge['date'] = DC_merge['GMT Time'].dt.date

# %%

DC_merge = pd.merge(DC_merge, n2ex, on=['date', 'EFA Block'], how='left')
DC_merge = pd.merge(DC_merge, epex, on=['date', 'EFA Block'], how='left')

#%%

DC_merge['interaction'] = DC_merge['Volume Requirements Forecast - DC-H - GB (MW)'] * DC_merge['Day Ahead Price (N2EX, local) - GB (£/MWh)']
DC_merge['interaction2'] = DC_merge['Volume Requirements Forecast - DC-H - GB (MW)'] * DC_merge['Day Ahead Price (EPEX, local) - GB (£/MWh)']

#%%




#%%

plt.scatter(DC_merge['interaction'], DC_merge['Ancillary Price - DC-H - GB (£/MW/h)'])
plt.xlabel('Volume Requirements Forecast - DC-H - GB (MW) * Day Ahead Price (N2EX, local) - GB (£/MWh)')
plt.ylabel('Ancillary Price - DC-H - GB (£/MW/h)')
plt.title('Volume Requirements Forecast vs Ancillary Price for DC-H')
plt.grid()
plt.tight_layout()
plt.show()

#%%

plt.scatter(DC_merge['interaction2'], DC_merge['Ancillary Price - DC-H - GB (£/MW/h)'])
plt.xlabel('Volume Requirements Forecast - DC-H - GB (MW) * Day Ahead Price (N2EX, local) - GB (£/MWh)')
plt.ylabel('Ancillary Price - DC-H - GB (£/MW/h)')
plt.title('Volume Requirements Forecast vs Ancillary Price for DC-H')
plt.grid()
plt.tight_layout()
plt.show()

# %%

plt.scatter(DC_merge['Volume Requirements Forecast - DC-H - GB (MW)'], DC_merge['Day Ahead Price (EPEX, local) - GB (£/MWh)'])
plt.xlabel('Volume Requirements Forecast - DC-H - GB (MW)') 
plt.ylabel(['Day Ahead Price (EPEX, local) - GB (£/MWh)'])
plt.title('Volume Requirements Forecast vs Ancillary Price for DC-H')
plt.grid()
plt.tight_layout()
plt.show()

#%%

plt.scatter(DC_merge['Day Ahead Price (N2EX, local) - GB (£/MWh)'], DC_merge['Ancillary Price - DC-H - GB (£/MW/h)'])
plt.xlabel('Day Ahead Price (N2EX, local) - GB (£/MWh)')
plt.ylabel('Ancillary Price - DC-H - GB (£/MW/h)')
plt.title('Day Ahead Price vs Ancillary Price for DC-H')
plt.grid()
plt.tight_layout()
plt.show()

# %%

plt.scatter(DC_merge['Day Ahead Price (EPEX, local) - GB (£/MWh)'], DC_merge['Ancillary Price - DC-H - GB (£/MW/h)'])
plt.xlabel('Day Ahead Price (EPEX, local) - GB (£/MWh)')
plt.ylabel('Ancillary Price - DC-H - GB (£/MW/h)')
plt.title('Day Ahead Price vs Ancillary Price for DC-H')
plt.grid()
plt.tight_layout()
plt.show()

# %%

hh = df.copy()
hh['GMT Time'] = pd.to_datetime(hh['GMT Time'], dayfirst=True)
hh['date'] = hh['GMT Time'].apply(ef.shift_efa).dt.date
hh['EFA Block'] = hh['GMT Time'].dt.hour.apply(ef.get_efa_block)
hh['interaction'] = hh['National Demand Forecast (NDF) - GB (MW)'] * hh['Day Ahead Price (EPEX half-hourly, local) - GB (£/MWh)']
ndf = hh.groupby(['date', 'EFA Block'])['National Demand Forecast (NDF) - GB (MW)'].mean().reset_index()
# ndf = hh.groupby(['date', 'EFA Block'])['interaction'].mean().reset_index()

# %%

DC_merge = DC.copy()
DC_merge['EFA Block'] = DC_merge['GMT Time'].dt.hour.apply(ef.get_efa_block2)
DC_merge['date'] = DC_merge['GMT Time'].dt.date

DC_merge = pd.merge(DC_merge, ndf, on=['date', 'EFA Block'], how='left')

plt.scatter(DC_merge['National Demand Forecast (NDF) - GB (MW)'], DC_merge['Ancillary Price - DC-H - GB (£/MW/h)'])
plt.xlabel('National Demand Forecast (NDF) - GB (MW)')
plt.ylabel('Ancillary Price - DC-H - GB (£/MW/h)')
plt.title('National Demand Forecast vs Ancillary Price for DC-H')
plt.grid()
plt.tight_layout()
plt.show()

# %%

# PLOT AUTOCORRELATION FUNCTION for DC[['GMT TIME', 'Ancillary Price - DC-H - GB (£/MW/h)']]
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(DC['Ancillary Price - DC-H - GB (£/MW/h)'], lags=50)

# %%

gbf = pd.read_csv('2023\Generation by Fuel.csv')
MEL = pd.read_csv('2023\MEL below PN.csv')
freq = pd.read_csv('2023\Frequency.csv')

# %%

data = df.copy()
data['GMT Time'] = pd.to_datetime(data['GMT Time'], dayfirst=True)
data['Month'] = data['GMT Time'].dt.month
data['EFA Block'] = data['GMT Time'].dt.hour.apply(ef.get_efa_block)
data['date'] = data['GMT Time'].apply(ef.shift_efa).dt.date

# %%

avg_dem = data.groupby(['Month', 'EFA Block'])['National Demand Forecast (NDF) - GB (MW)'].mean().reset_index().rename(columns={'National Demand Forecast (NDF) - GB (MW)': 'Avg'})
std_dem = data.groupby(['Month', 'EFA Block'])['National Demand Forecast (NDF) - GB (MW)'].std().reset_index().rename(columns={'National Demand Forecast (NDF) - GB (MW)': 'Std'})
values = pd.merge(avg_dem, std_dem, on=['Month', 'EFA Block'])

#%%

data = pd.merge(data, values, left_on=['Month', 'EFA Block'], right_on=['Month', 'EFA Block'])
data['GMT Time'] = pd.to_datetime(data['GMT Time'], dayfirst=True)
data.sort_values(by='GMT Time', inplace=True)
data.reset_index(drop=True, inplace=True)

#%%

data['Demand Z-Score'] = (data['National Demand Forecast (NDF) - GB (MW)'] - data['Avg']) / data['Std']
data['Drift'] = data['National Demand Forecast (NDF) - GB (MW)'] - data['Avg']

# %%

drift = data.groupby(['date', 'EFA Block'])[['Drift', 'Demand Z-Score']].mean().reset_index()

#%%

def efablocktotime(date, efablock):
    if efablock == 'EFA 1':
        # return 23:00 time on that day
        return date + pd.Timedelta(hours=23)
    elif efablock == 'EFA 2':
        return date + pd.Timedelta(hours=3)
    elif efablock == 'EFA 3':
        return date + pd.Timedelta(hours=7)
    elif efablock == 'EFA 4':
        return date + pd.Timedelta(hours=11)
    elif efablock == 'EFA 5':
        return date + pd.Timedelta(hours=15)
    elif efablock == 'EFA 6':
        return date + pd.Timedelta(hours=19)

drift['date'] = pd.to_datetime(drift['date'])
drift['GMT Time'] = drift.apply(lambda x: efablocktotime(x['date'], x['EFA Block']), axis=1)
drift = drift[['GMT Time', 'Drift', 'Demand Z-Score']]
drift.to_csv('drift.csv')

# %%

DC_merge = DC.copy()
DC_merge['EFA Block'] = DC_merge['GMT Time'].dt.hour.apply(ef.get_efa_block2)
DC_merge['date'] = DC_merge['GMT Time'].dt.date
DC_merge = pd.merge(DC_merge, drift, on=['date', 'EFA Block'], how='left')

# %%

# give me the r^2 value 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
X = DC_merge[['Drift']]
y = DC_merge['Ancillary Price - DC-H - GB (£/MW/h)']
model = LinearRegression()
# give me the r2
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f'R^2: {r2}')

plt.scatter(DC_merge['Drift'], DC_merge['Ancillary Price - DC-H - GB (£/MW/h)'])
plt.xlabel('Drift') 
plt.ylabel('Ancillary Price - DC-H - GB (£/MW/h)')
plt.title('Drift vs Ancillary Price for DC-H')
plt.grid()
plt.tight_layout()
plt.show()

# %%

'''
Checking correlation of interaction terms 
'''

