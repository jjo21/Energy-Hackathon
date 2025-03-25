#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_exploration_funcs as ef

#%%

avp = pd.read_csv('2023\Ancillary Volumes & Prices (4H).csv')
dap = pd.read_csv('2023\Day-Ahead Price (1H).csv').dropna()
df = pd.read_csv('2023\Prices & Forecasts (HH).csv')

avp2 = pd.read_csv('2024\Ancillary Volumes & Prices (4H).csv')
dap2 = pd.read_csv('2024\Day-Ahead Price (1H).csv')
df2 = pd.read_csv('2024\Prices & Forecasts (HH).csv')

# %%

paf = df.copy()
paf['GMT Time'] = pd.to_datetime(paf['GMT Time'])
paf['Hour'] = paf['GMT Time'].dt.hour
paf['Day'] = paf['GMT Time'].dt.month
paf['EFA Block'] = paf['Hour'].apply(ef.get_efa_block)

# %%

plt.scatter(df['National Demand Forecast (NDF) - GB (MW)'], df['Day Ahead Price (EPEX half-hourly, local) - GB (£/MWh)'])
plt.xlabel('National Demand Forecast (NDF) - GB (MW)')
plt.ylabel('Day Ahead Price (EPEX half-hourly, local) - GB (£/MWh)')
plt.title('National Demand Forecast vs Day Ahead Price')
plt.grid()
plt.tight_layout()
plt.savefig('Plots/2023 National Demand Forecast vs Day Ahead Price.png')
plt.show()

# %%

# Plotting the scatter plot for national demand forecast against day ahead price

for i in range(6):
    hold = paf[paf['EFA Block'] == f'EFA {i+1}'].copy()
    plt.scatter(hold['National Demand Forecast (NDF) - GB (MW)'], hold['Day Ahead Price (EPEX half-hourly, local) - GB (£/MWh)'])
    plt.xlabel('National Demand Forecast (NDF) - GB (MW)')
    plt.ylabel('Day Ahead Price (EPEX half-hourly, local) - GB (£/MWh)')
    plt.title(f'National Demand Forecast vs Day Ahead Price for EFA {i+1}')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'Plots/2023 National Demand Forecast vs Day Ahead Price for EFA {i+1}.png')
    plt.show()

# %%

# Plotting the histogram for national demand forecast

for i in range(6):
    hold = paf[paf['EFA Block'] == f'EFA {i+1}'].copy()
    plt.hist(hold['National Demand Forecast (NDF) - GB (MW)'], bins=20, alpha=0.7, label=f'EFA {i+1}') 
    plt.xlabel('National Demand Forecast (NDF) - GB (MW)')
    plt.ylabel('Frequency')
    plt.title(f'National Demand Forecast for EFA {i+1}')
    plt.grid()
    plt.tight_layout()

plt.title(f'National Demand Forecast')
plt.legend()
plt.savefig(f'Plots/2023 National Demand Forecast.png')
plt.show()

# %%

# Plotting how statistics change with each EFA block 
mean = []
std = []
skew = []
kurt = []
EFA_blocks = ['EFA 1', 'EFA 2', 'EFA 3', 'EFA 4', 'EFA 5', 'EFA 6']

for i in range(6):
    hold = paf[paf['EFA Block'] == f'EFA {i+1}'].copy()
    mean.append(hold['National Demand Forecast (NDF) - GB (MW)'].mean())
    std.append(hold['National Demand Forecast (NDF) - GB (MW)'].std())
    skew.append(hold['National Demand Forecast (NDF) - GB (MW)'].skew())
    kurt.append(hold['National Demand Forecast (NDF) - GB (MW)'].kurt())

plt.plot(EFA_blocks, mean, label='Mean')
plt.xlabel('EFA Blocks')
plt.ylabel('National Demand Forecast (NDF) - GB (MW)')
plt.title('Mean for National Demand Forecast')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('Plots/2023 Mean for National Demand Forecast.png')
plt.show()
#%%

plt.plot(EFA_blocks, std, label='Standard Deviation')
plt.xlabel('EFA Blocks')
plt.ylabel('National Demand Forecast (NDF) - GB (MW)')
plt.title('Standard Deviation for National Demand Forecast')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('Plots/2023 Standard Deviation for National Demand Forecast.png')
plt.show()

# %%

plt.plot(EFA_blocks, skew, label='Skewness')
plt.plot(EFA_blocks, kurt, label='Kurtosis')
plt.xlabel('EFA Blocks')
plt.ylabel('Statistics')
plt.title('Statistics for National Demand Forecast')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('Plots/Skew and Kurtosis for National Demand Forecast.png')
plt.show()

# %%

# do the same for price 

for i in range(6):
    hold = paf[paf['EFA Block'] == f'EFA {i+1}'].copy()
    plt.hist(hold['Day Ahead Price (EPEX half-hourly, local) - GB (£/MWh)'], bins=20, alpha=0.7, label=f'EFA {i+1}')
    plt.xlabel('Day Ahead Price (EPEX half-hourly, local) - GB (£/MWh)')
    plt.ylabel('Frequency')
    plt.title(f'Day Ahead Price for EFA {i+1}')
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig('Plots/2023 Day Ahead Price.png')
plt.show()

# %%

mean = []
std = []
skew = []
kurt = []

for i in range(6):
    hold = paf[paf['EFA Block'] == f'EFA {i+1}'].copy()
    mean.append(hold['Day Ahead Price (EPEX half-hourly, local) - GB (£/MWh)'].mean())
    std.append(hold['Day Ahead Price (EPEX half-hourly, local) - GB (£/MWh)'].std())
    skew.append(hold['Day Ahead Price (EPEX half-hourly, local) - GB (£/MWh)'].skew())
    kurt.append(hold['Day Ahead Price (EPEX half-hourly, local) - GB (£/MWh)'].kurt())

plt.plot(EFA_blocks, mean, label='Mean')
plt.xlabel('EFA Blocks')
plt.ylabel('Day Ahead Price (EPEX half-hourly, local) - GB (£/MWh)')
plt.title('Mean for Day Ahead Price')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('Plots/2023 Mean for Day Ahead Price.png')
plt.show()

# %%

plt.plot(EFA_blocks, std, label='Standard Deviation')
plt.xlabel('EFA Blocks')
plt.ylabel('Day Ahead Price (EPEX half-hourly, local) - GB (£/MWh)')
plt.title('Standard Deviation for Day Ahead Price')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('Plots/2023 Standard Deviation for Day Ahead Price.png')
plt.show()

# %%

plt.plot(EFA_blocks, skew, label='Skewness')
plt.plot(EFA_blocks, kurt, label='Kurtosis')
plt.xlabel('EFA Blocks')
plt.ylabel('Statistics')
plt.title('Statistics for Day Ahead Price')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('Plots/Skew and Kurtosis for Day Ahead Price.png')
plt.show()

# %%
