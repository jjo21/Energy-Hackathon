#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import kent_functions as kf

importlib.reload(kf)

avp = pd.read_csv('2023\Ancillary Volumes & Prices (4H).csv')
dap = pd.read_csv('2023\Day-Ahead Price (1H).csv').dropna()
df1 = pd.read_csv('2023\Prices & Forecasts (HH).csv')
freq1 = pd.read_csv('2023\Frequency.csv')
gen1 = pd.read_csv('2023\Generation by Fuel.csv')
mel1 = pd.read_csv('2023\MEL below PN.csv')

avp2 = pd.read_csv('2024\Ancillary Volumes & Prices (4H).csv')
dap2 = pd.read_csv('2024\Day-Ahead Price (1H).csv')
df2 = pd.read_csv('2024\Prices & Forecasts (HH).csv')
freq2 = pd.read_csv('2024\Frequency.csv')
gen2 = pd.read_csv('2024\Generation by Fuel.csv')
mel2 = pd.read_csv('2024\MEL below PN.csv')

freq2['GMT Time'] = freq2['GMT Time'].apply(kf.parse_excel_or_date)
freq1['GMT Time'] = pd.to_datetime(freq1['GMT Time'], dayfirst=True)

tavp = pd.concat([avp, avp2], ignore_index=True)
tavp['GMT Time'] = pd.to_datetime(tavp['GMT Time'], dayfirst=True)
tavp['EFA'] = tavp['GMT Time'].apply(kf.get_efa_block)

tdap = pd.concat([dap, dap2], ignore_index=True)
tdap['GMT Time'] = pd.to_datetime(tdap['GMT Time'], dayfirst=True)
tdap['EFA'] = tdap['GMT Time'].apply(kf.get_efa_block)

paf = pd.concat([df1, df2], ignore_index=True)
paf['GMT Time'] = pd.to_datetime(paf['GMT Time'], dayfirst=True)
tavp['EFA'] = tavp['GMT Time'].apply(kf.get_efa_block)

freq = pd.concat([freq1, freq2], ignore_index=True)

gen = pd.concat([gen1, gen2], ignore_index=True)
gen['GMT Time'] = pd.to_datetime(gen['GMT Time'], dayfirst=True)

mel = pd.concat([mel1, mel2], ignore_index=True)
mel['GMT Time'] = pd.to_datetime(mel['GMT Time'], dayfirst=True)

#%%
aggregated_ndf = kf.aggregate_all(paf, 'National Demand Forecast (NDF) - GB (MW)') #aggregated NDF using different methods
aggregated_EPEX = kf.aggregate_all(tdap, 'Day Ahead Price (EPEX, local) - GB (£/MWh)') #aggregated EPEX using different methods
aggregated_N2EX = kf.aggregate_all(tdap, 'Day Ahead Price (N2EX, local) - GB (£/MWh)') #aggregated N2EX using different methods

mean_freq = kf.aggregate(freq, 'Rolling System Frequency - GB (Hz) - Half-Hour Average', 'mean') #aggregated frequency using mean method
max_freq = kf.aggregate(freq, 'Rolling System Frequency - GB (Hz) - Half-Hour Maximum', 'max') #aggregated frequency using max method
min_freq = kf.aggregate(freq, 'Rolling System Frequency - GB (Hz) - Half-Hour Minimum', 'min') #aggregated frequency using min method
aggregated_freq = pd.concat([mean_freq, max_freq, min_freq], axis=1).reset_index() #concatenate all frequency dataframes

aggregated_ndf['EFA'] = aggregated_ndf['GMT Time'].apply(kf.get_efa_block)
aggregated_EPEX['EFA'] = aggregated_EPEX['GMT Time'].apply(kf.get_efa_block)
aggregated_N2EX['EFA'] = aggregated_N2EX['GMT Time'].apply(kf.get_efa_block)

volume_labels = [f'Vol {x}' for x in ['DC-H', 'DC-L', 'DR-H', 'DR-L', 'DM-H', 'DM-L']]
price_labels = [f'Price {x}' for x in ['DC-H', 'DC-L', 'DR-H', 'DR-L', 'DM-H', 'DM-L']] 

mtavp = tavp.copy()
maggregated_ndf = aggregated_ndf.copy()
maggregated_EPEX = aggregated_EPEX.copy()
maggregated_N2EX = aggregated_N2EX.copy()

#no data splitting
kf.corr_grid(mtavp, mtavp, mtavp.columns[1:7], mtavp.columns[13:19], volume_labels, price_labels, header="Volume Forecasts vs Ancillary Prices", filename='plots_kent/volume_price_grid.png')
kf.corr_grid(maggregated_ndf, mtavp, maggregated_ndf.columns[1:7], mtavp.columns[1:7], maggregated_ndf.columns[1:7], volume_labels, header="Aggregated NDF vs Volume Forecasts", filename='plots_kent/ndf_volume_grid.png')
kf.corr_grid(maggregated_ndf, mtavp, maggregated_ndf.columns[1:7], mtavp.columns[13:19], maggregated_ndf.columns[1:7], price_labels, header="Aggregated NDF vs Ancillary Prices", filename='plots_kent/ndf_price_grid.png')
kf.corr_grid(maggregated_EPEX, mtavp, maggregated_EPEX.columns[1:7], mtavp.columns[1:7], maggregated_EPEX.columns[1:7], volume_labels, header="Aggregated EPEX Day-Ahead Price vs Volume Forecasts", filename='plots_kent/epex_volume_grid.png')
kf.corr_grid(maggregated_EPEX, mtavp, maggregated_EPEX.columns[1:7], mtavp.columns[13:19], maggregated_EPEX.columns[1:7], price_labels, header="Aggregated EPEX Day-Ahead Price vs Ancillary Prices", filename='plots_kent/epex_price_grid.png')
kf.corr_grid(maggregated_N2EX, mtavp, maggregated_N2EX.columns[1:7], mtavp.columns[1:7], maggregated_N2EX.columns[1:7], volume_labels, header="Aggregated N2EX Day-Ahead Price vs Volume Forecasts", filename='plots_kent/n2ex_volume_grid.png')
kf.corr_grid(maggregated_N2EX, mtavp, maggregated_N2EX.columns[1:7], mtavp.columns[13:19], maggregated_N2EX.columns[1:7], price_labels, header="Aggregated N2EX Day-Ahead Price vs Ancillary Prices", filename='plots_kent/n2ex_price_grid.png')

#lagged variables
kf.corr_grid(mtavp, mtavp, mtavp.columns[7:13], mtavp.columns[13:19], volume_labels, price_labels, header="Volume Accepted vs Ancillary Prices", filename='plots_kent/accepted_volume_price_grid.png')

#post EAC
eactavp = tavp[tavp['GMT Time'] >= pd.Timestamp('2023-11-03')].copy()
eacaggregated_ndf = aggregated_ndf[aggregated_ndf['GMT Time'] >= pd.Timestamp('2023-11-03')].copy()
eacaggregated_EPEX = aggregated_EPEX[aggregated_EPEX['GMT Time'] >= pd.Timestamp('2023-11-03')].copy()
eacaggregated_N2EX = aggregated_N2EX[aggregated_N2EX['GMT Time'] >= pd.Timestamp('2023-11-03')].copy()

kf.corr_grid(eactavp, eactavp, eactavp.columns[1:7], eactavp.columns[13:19], volume_labels, price_labels, header="Post EAC Volume Forecasts vs Ancillary Prices", filename='plots_kent/post_eac_volume_price_grid.png')
kf.corr_grid(eacaggregated_ndf, eactavp, eacaggregated_ndf.columns[1:7], eactavp.columns[1:7], eacaggregated_ndf.columns[1:7], volume_labels, header="Post EAC Aggregated NDF vs Volume Forecasts", filename='plots_kent/post_eac_ndf_volume_grid.png')
kf.corr_grid(eacaggregated_ndf, eactavp, eacaggregated_ndf.columns[1:7], eactavp.columns[13:19], eacaggregated_ndf.columns[1:7], price_labels, header="Post EAC Aggregated NDF vs Ancillary Prices", filename='plots_kent/post_eac_ndf_price_grid.png')
kf.corr_grid(eacaggregated_EPEX, eactavp, eacaggregated_EPEX.columns[1:7], eactavp.columns[1:7], eacaggregated_EPEX.columns[1:7], volume_labels, header="Post EAC Aggregated EPEX Day-Ahead Price vs Volume Forecasts", filename='plots_kent/post_eac_epex_volume_grid.png')
kf.corr_grid(eacaggregated_EPEX, eactavp, eacaggregated_EPEX.columns[1:7], eactavp.columns[13:19], eacaggregated_EPEX.columns[1:7], price_labels, header="Post EAC Aggregated EPEX Day-Ahead Price vs Ancillary Prices", filename='plots_kent/post_eac_epex_price_grid.png')
kf.corr_grid(eacaggregated_N2EX, eactavp, eacaggregated_N2EX.columns[1:7], eactavp.columns[1:7], eacaggregated_N2EX.columns[1:7], volume_labels, header="Post EAC Aggregated N2EX Day-Ahead Price vs Volume Forecasts", filename='plots_kent/post_eac_n2ex_volume_grid.png')
kf.corr_grid(eacaggregated_N2EX, eactavp, eacaggregated_N2EX.columns[1:7], eactavp.columns[13:19], eacaggregated_N2EX.columns[1:7], price_labels, header="Post EAC Aggregated N2EX Day-Ahead Price vs Ancillary Prices", filename='plots_kent/post_eac_n2ex_price_grid.png')

#lagged variables
kf.corr_grid(eactavp, eactavp, eactavp.columns[7:13], eactavp.columns[13:19], volume_labels, price_labels, header="Post EAC Volume Accepted vs Ancillary Prices", filename='plots_kent/post_eac_accepted_volume_price_grid.png')

#efa block 1
mtavp = eactavp[eactavp['EFA'] == 1].copy()
maggregated_ndf = eacaggregated_ndf[eacaggregated_ndf['EFA'] == 1].copy()
maggregated_EPEX = eacaggregated_EPEX[eacaggregated_EPEX['EFA'] == 1].copy()
maggregated_N2EX = eacaggregated_N2EX[eacaggregated_N2EX['EFA'] == 1].copy()

kf.corr_grid(mtavp, mtavp, mtavp.columns[1:7], mtavp.columns[13:19], volume_labels, price_labels, header="EFA1 Volume Forecasts vs Ancillary Prices", filename='plots_kent/efa1_volume_price_grid.png')
kf.corr_grid(maggregated_ndf, mtavp, maggregated_ndf.columns[1:7], mtavp.columns[1:7], maggregated_ndf.columns[1:7], volume_labels, header="EFA1 Aggregated NDF vs Volume Forecasts", filename='plots_kent/efa1_ndf_volume_grid.png')
kf.corr_grid(maggregated_ndf, mtavp, maggregated_ndf.columns[1:7], mtavp.columns[13:19], maggregated_ndf.columns[1:7], price_labels, header="EFA1 Aggregated NDF vs Ancillary Prices", filename='plots_kent/efa1_ndf_price_grid.png')
kf.corr_grid(maggregated_EPEX, mtavp, maggregated_EPEX.columns[1:7], mtavp.columns[1:7], maggregated_EPEX.columns[1:7], volume_labels, header="EFA1 Aggregated EPEX Day-Ahead Price vs Volume Forecasts", filename='plots_kent/efa1_epex_volume_grid.png')
kf.corr_grid(maggregated_EPEX, mtavp, maggregated_EPEX.columns[1:7], mtavp.columns[13:19], maggregated_EPEX.columns[1:7], price_labels, header="EFA1 Aggregated EPEX Day-Ahead Price vs Ancillary Prices", filename='plots_kent/efa1_epex_price_grid.png')
kf.corr_grid(maggregated_N2EX, mtavp, maggregated_N2EX.columns[1:7], mtavp.columns[1:7], maggregated_N2EX.columns[1:7], volume_labels, header="EFA1 Aggregated N2EX Day-Ahead Price vs Volume Forecasts", filename='plots_kent/efa1_n2ex_volume_grid.png')
kf.corr_grid(maggregated_N2EX, mtavp, maggregated_N2EX.columns[1:7], mtavp.columns[13:19], maggregated_N2EX.columns[1:7], price_labels, header="EFA1 Aggregated N2EX Day-Ahead Price vs Ancillary Prices", filename='plots_kent/efa1_n2ex_price_grid.png')

#lagged variables
kf.corr_grid(mtavp, mtavp, mtavp.columns[7:13], mtavp.columns[13:19], volume_labels, price_labels, header="EFA1 Volume Accepted vs Ancillary Prices", filename='plots_kent/efa1_accepted_volume_price_grid.png')

#quarter 1
mtavp = eactavp[eactavp['GMT Time'].dt.quarter == 1].copy()
maggregated_ndf = eacaggregated_ndf[eacaggregated_ndf['GMT Time'].dt.quarter == 1].copy()
maggregated_EPEX = eacaggregated_EPEX[eacaggregated_EPEX['GMT Time'].dt.quarter == 1].copy()
maggregated_N2EX = eacaggregated_N2EX[eacaggregated_N2EX['GMT Time'].dt.quarter == 1].copy()

kf.corr_grid(mtavp, mtavp, mtavp.columns[1:7], mtavp.columns[13:19], volume_labels, price_labels, header="Q1 Volume Forecasts vs Ancillary Prices", filename='plots_kent/q1_volume_price_grid.png')
kf.corr_grid(maggregated_ndf, mtavp, maggregated_ndf.columns[1:7], mtavp.columns[1:7], maggregated_ndf.columns[1:7], volume_labels, header="Q1 Aggregated NDF vs Volume Forecasts", filename='plots_kent/q1_ndf_volume_grid.png')
kf.corr_grid(maggregated_ndf, mtavp, maggregated_ndf.columns[1:7], mtavp.columns[13:19], maggregated_ndf.columns[1:7], price_labels, header="Q1 Aggregated NDF vs Ancillary Prices", filename='plots_kent/q1_ndf_price_grid.png')
kf.corr_grid(maggregated_EPEX, mtavp, maggregated_EPEX.columns[1:7], mtavp.columns[1:7], maggregated_EPEX.columns[1:7], volume_labels, header="Q1 Aggregated EPEX Day-Ahead Price vs Volume Forecasts", filename='plots_kent/q1_epex_volume_grid.png')
kf.corr_grid(maggregated_EPEX, mtavp, maggregated_EPEX.columns[1:7], mtavp.columns[13:19], maggregated_EPEX.columns[1:7], price_labels, header="Q1 Aggregated EPEX Day-Ahead Price vs Ancillary Prices", filename='plots_kent/q1_epex_price_grid.png')
kf.corr_grid(maggregated_N2EX, mtavp, maggregated_N2EX.columns[1:7], mtavp.columns[1:7], maggregated_N2EX.columns[1:7], volume_labels, header="Q1 Aggregated N2EX Day-Ahead Price vs Volume Forecasts", filename='plots_kent/q1_n2ex_volume_grid.png')
kf.corr_grid(maggregated_N2EX, mtavp, maggregated_N2EX.columns[1:7], mtavp.columns[13:19], maggregated_N2EX.columns[1:7], price_labels, header="Q1 Aggregated N2EX Day-Ahead Price vs Ancillary Prices", filename='plots_kent/q1_n2ex_price_grid.png')

#lagged variables
kf.corr_grid(mtavp, mtavp, mtavp.columns[7:13], mtavp.columns[13:19], volume_labels, price_labels, header="Q1 Volume Accepted vs Ancillary Prices", filename='plots_kent/q1_accepted_volume_price_grid.png')

#weekend
mtavp = eactavp[eactavp['GMT Time'].dt.dayofweek >= 5].copy()
maggregated_ndf = eacaggregated_ndf[eacaggregated_ndf['GMT Time'].dt.dayofweek >= 5].copy()
maggregated_EPEX = eacaggregated_EPEX[eacaggregated_EPEX['GMT Time'].dt.dayofweek >= 5].copy()
maggregated_N2EX = eacaggregated_N2EX[eacaggregated_N2EX['GMT Time'].dt.dayofweek >= 5].copy()

kf.corr_grid(mtavp, mtavp, mtavp.columns[1:7], mtavp.columns[13:19], volume_labels, price_labels, header="Weekend Volume Forecasts vs Ancillary Prices", filename='plots_kent/wknd_volume_price_grid.png')
kf.corr_grid(maggregated_ndf, mtavp, maggregated_ndf.columns[1:7], mtavp.columns[1:7], maggregated_ndf.columns[1:7], volume_labels, header="Weekend Aggregated NDF vs Volume Forecasts", filename='plots_kent/wknd_ndf_volume_grid.png')
kf.corr_grid(maggregated_ndf, mtavp, maggregated_ndf.columns[1:7], mtavp.columns[13:19], maggregated_ndf.columns[1:7], price_labels, header="Weekend Aggregated NDF vs Ancillary Prices", filename='plots_kent/wknd_ndf_price_grid.png')
kf.corr_grid(maggregated_EPEX, mtavp, maggregated_EPEX.columns[1:7], mtavp.columns[1:7], maggregated_EPEX.columns[1:7], volume_labels, header="Weekend Aggregated EPEX Day-Ahead Price vs Volume Forecasts", filename='plots_kent/wknd_epex_volume_grid.png')
kf.corr_grid(maggregated_EPEX, mtavp, maggregated_EPEX.columns[1:7], mtavp.columns[13:19], maggregated_EPEX.columns[1:7], price_labels, header="Weekend Aggregated EPEX Day-Ahead Price vs Ancillary Prices", filename='plots_kent/wknd_epex_price_grid.png')
kf.corr_grid(maggregated_N2EX, mtavp, maggregated_N2EX.columns[1:7], mtavp.columns[1:7], maggregated_N2EX.columns[1:7], volume_labels, header="Weekend Aggregated N2EX Day-Ahead Price vs Volume Forecasts", filename='plots_kent/wknd_n2ex_volume_grid.png')
kf.corr_grid(maggregated_N2EX, mtavp, maggregated_N2EX.columns[1:7], mtavp.columns[13:19], maggregated_N2EX.columns[1:7], price_labels, header="Weekend Aggregated N2EX Day-Ahead Price vs Ancillary Prices", filename='plots_kent/wknd_n2ex_price_grid.png')

#lagged variables
kf.corr_grid(mtavp, mtavp, mtavp.columns[7:13], mtavp.columns[13:19], volume_labels, price_labels, header="Volume Accepted vs Ancillary Prices", filename='plots_kent/accepted_volume_price_grid.png')

#weekday
mtavp = eactavp[eactavp['GMT Time'].dt.dayofweek < 5].copy()
maggregated_ndf = eacaggregated_ndf[eacaggregated_ndf['GMT Time'].dt.dayofweek < 5].copy()
maggregated_EPEX = eacaggregated_EPEX[eacaggregated_EPEX['GMT Time'].dt.dayofweek < 5].copy()
maggregated_N2EX = eacaggregated_N2EX[eacaggregated_N2EX['GMT Time'].dt.dayofweek < 5].copy()

kf.corr_grid(mtavp, mtavp, mtavp.columns[1:7], mtavp.columns[13:19], volume_labels, price_labels, header="Weekday Volume Forecasts vs Ancillary Prices", filename='plots_kent/wkdy_volume_price_grid.png')
kf.corr_grid(maggregated_ndf, mtavp, maggregated_ndf.columns[1:7], mtavp.columns[1:7], maggregated_ndf.columns[1:7], volume_labels, header="Weekday Aggregated NDF vs Volume Forecasts", filename='plots_kent/wkdy_ndf_volume_grid.png')
kf.corr_grid(maggregated_ndf, mtavp, maggregated_ndf.columns[1:7], mtavp.columns[13:19], maggregated_ndf.columns[1:7], price_labels, header="Weekday Aggregated NDF vs Ancillary Prices", filename='plots_kent/wkdy_ndf_price_grid.png')
kf.corr_grid(maggregated_EPEX, mtavp, maggregated_EPEX.columns[1:7], mtavp.columns[1:7], maggregated_EPEX.columns[1:7], volume_labels, header="Weekday Aggregated EPEX Day-Ahead Price vs Volume Forecasts", filename='plots_kent/wkdy_epex_volume_grid.png')
kf.corr_grid(maggregated_EPEX, mtavp, maggregated_EPEX.columns[1:7], mtavp.columns[13:19], maggregated_EPEX.columns[1:7], price_labels, header="Weekday Aggregated EPEX Day-Ahead Price vs Ancillary Prices", filename='plots_kent/wkdy_epex_price_grid.png')
kf.corr_grid(maggregated_N2EX, mtavp, maggregated_N2EX.columns[1:7], mtavp.columns[1:7], maggregated_N2EX.columns[1:7], volume_labels, header="Weekday Aggregated N2EX Day-Ahead Price vs Volume Forecasts", filename='plots_kent/wkdy_n2ex_volume_grid.png')
kf.corr_grid(maggregated_N2EX, mtavp, maggregated_N2EX.columns[1:7], mtavp.columns[13:19], maggregated_N2EX.columns[1:7], price_labels, header="Weekday Aggregated N2EX Day-Ahead Price vs Ancillary Prices", filename='plots_kent/wkdy_n2ex_price_grid.png')

#lagged variables
kf.corr_grid(mtavp, mtavp, mtavp.columns[7:13], mtavp.columns[13:19], volume_labels, price_labels, header="Weekday Volume Accepted vs Ancillary Prices", filename='plots_kent/wkdy_accepted_volume_price_grid.png')

#fridays
mtavp = eactavp[eactavp['GMT Time'].dt.dayofweek == 4].copy()
maggregated_ndf = eacaggregated_ndf[eacaggregated_ndf['GMT Time'].dt.dayofweek == 4].copy()
maggregated_EPEX = eacaggregated_EPEX[eacaggregated_EPEX['GMT Time'].dt.dayofweek == 4].copy()
maggregated_N2EX = eacaggregated_N2EX[eacaggregated_N2EX['GMT Time'].dt.dayofweek == 4].copy()

kf.corr_grid(mtavp, mtavp, mtavp.columns[1:7], mtavp.columns[13:19], volume_labels, price_labels, header="Friday Volume Forecasts vs Ancillary Prices", filename='plots_kent/fri_volume_price_grid.png')
kf.corr_grid(maggregated_ndf, mtavp, maggregated_ndf.columns[1:7], mtavp.columns[1:7], maggregated_ndf.columns[1:7], volume_labels, header="Friday Aggregated NDF vs Volume Forecasts", filename='plots_kent/fri_ndf_volume_grid.png')
kf.corr_grid(maggregated_ndf, mtavp, maggregated_ndf.columns[1:7], mtavp.columns[13:19], maggregated_ndf.columns[1:7], price_labels, header="Friday Aggregated NDF vs Ancillary Prices", filename='plots_kent/fri_ndf_price_grid.png')
kf.corr_grid(maggregated_EPEX, mtavp, maggregated_EPEX.columns[1:7], mtavp.columns[1:7], maggregated_EPEX.columns[1:7], volume_labels, header="Friday Aggregated EPEX Day-Ahead Price vs Volume Forecasts", filename='plots_kent/fri_epex_volume_grid.png')
kf.corr_grid(maggregated_EPEX, mtavp, maggregated_EPEX.columns[1:7], mtavp.columns[13:19], maggregated_EPEX.columns[1:7], price_labels, header="Friday Aggregated EPEX Day-Ahead Price vs Ancillary Prices", filename='plots_kent/fri_epex_price_grid.png')
kf.corr_grid(maggregated_N2EX, mtavp, maggregated_N2EX.columns[1:7], mtavp.columns[1:7], maggregated_N2EX.columns[1:7], volume_labels, header="Friday Aggregated N2EX Day-Ahead Price vs Volume Forecasts", filename='plots_kent/fri_n2ex_volume_grid.png')
kf.corr_grid(maggregated_N2EX, mtavp, maggregated_N2EX.columns[1:7], mtavp.columns[13:19], maggregated_N2EX.columns[1:7], price_labels, header="Friday Aggregated N2EX Day-Ahead Price vs Ancillary Prices", filename='plots_kent/fri_n2ex_price_grid.png')

#lagged variables
kf.corr_grid(mtavp, mtavp, mtavp.columns[7:13], mtavp.columns[13:19], volume_labels, price_labels, header="Friday Volume Accepted vs Ancillary Prices", filename='plots_kent/fri_accepted_volume_price_grid.png')

#next steps: 
# 1: lag plot for prices (maybe weekly, daily or by efa))

lags = pd.concat([eactavp.iloc[:, 0], eactavp.iloc[:, 13:19]], axis=1)
kf.lag_plot_matrix(lags, freq='EFA', max_lag=6, label=price_labels, header='Lag Plot by EFA Block', filename='plots_kent/lag_matrix_efa.png')
kf.lag_plot_matrix(lags, freq='D', max_lag=7, label=price_labels, header='Lag Plot by Day', filename='plots_kent/lag_matrix_day.png')
kf.lag_plot_matrix(lags, freq='W', max_lag=9, label=price_labels, header='Lag Plot by Week', filename='plots_kent/lag_matrix_week.png')
kf.lag_plot_matrix(lags, freq='Q', max_lag=4, label=price_labels, header='Lag Plot by Quarter', filename='plots_kent/lag_matrix_quarter.png')

# 2: matrix for lagged variables to decide which are worth forecasting (time series analysis)
#done for accepted volumes above. for other variables (e.g. freq, gen, mel etc, not done)

# 3: train models with non-lagged and lagged variables to decide which variables are worth forecasting (time series analysis)
# 4: can try time series analysis to forecast prices as well