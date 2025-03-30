#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kent_functions as kf
import seaborn as sns

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

#%%
volume_cols = tavp.columns[1:7]   # Columns 1–6: Volumes
price_cols = tavp.columns[13:19]  # Columns 13–18: Ancillary Prices
aggregated_ndf = kf.aggregate_all(paf, 'National Demand Forecast (NDF) - GB (MW)') #aggregated NDF using different methods
aggregated_EPEX = kf.aggregate_all(tdap, 'Day Ahead Price (EPEX, local) - GB (£/MWh)') #aggregated EPEX using different methods
aggregated_N2EX = kf.aggregate_all(tdap, 'Day Ahead Price (N2EX, local) - GB (£/MWh)') #aggregated N2EX using different methods

volume_labels = [f'Vol {x}' for x in ['DC-H', 'DC-L', 'DR-H', 'DR-L', 'DM-H', 'DM-L']]
price_labels = [f'Price {x}' for x in ['DC-H', 'DC-L', 'DR-H', 'DR-L', 'DM-H', 'DM-L']]

kf.corr_grid(tavp, tavp, volume_cols, price_cols, volume_labels, price_labels, header="Volume Forecasts vs Ancillary Prices", filename='plots_kent/volume_price_grid.png')
kf.corr_grid(aggregated_ndf, tavp, aggregated_ndf.columns[1:], volume_cols, aggregated_ndf.columns[1:], volume_labels, header="Aggregated NDF vs Volume Forecasts", filename='plots_kent/ndf_volume_grid.png')
kf.corr_grid(aggregated_ndf, tavp, aggregated_ndf.columns[1:], price_cols, aggregated_ndf.columns[1:], price_labels, header="Aggregated NDF vs Ancillary Prices", filename='plots_kent/ndf_price_grid.png')
kf.corr_grid(aggregated_EPEX, tavp, aggregated_EPEX.columns[1:], volume_cols, aggregated_EPEX.columns[1:], volume_labels, header="Aggregated EPEX Day-Ahead Price vs Volume Forecasts", filename='plots_kent/epex_volume_grid.png')
kf.corr_grid(aggregated_EPEX, tavp, aggregated_EPEX.columns[1:], price_cols, aggregated_EPEX.columns[1:], price_labels, header="Aggregated EPEX Day-Ahead Price vs Ancillary Prices", filename='plots_kent/epex_price_grid.png')
kf.corr_grid(aggregated_N2EX, tavp, aggregated_N2EX.columns[1:], volume_cols, aggregated_N2EX.columns[1:], volume_labels, header="Aggregated N2EX Day-Ahead Price vs Volume Forecasts", filename='plots_kent/n2ex_volume_grid.png')
kf.corr_grid(aggregated_N2EX, tavp, aggregated_N2EX.columns[1:], price_cols, aggregated_N2EX.columns[1:], price_labels, header="Aggregated N2EX Day-Ahead Price vs Ancillary Prices", filename='plots_kent/n2ex_price_grid.png')