#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_exploration_funcs as ef
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#%%

avp = pd.read_csv('2023\Ancillary Volumes & Prices (4H).csv')
avp2 = pd.read_csv('2024\Ancillary Volumes & Prices (4H).csv')
dap = pd.read_csv('2023\Day-Ahead Price (1H).csv').dropna()
dap2 = pd.read_csv('2024\Day-Ahead Price (1H).csv').dropna()
df = pd.read_csv('2023\Prices & Forecasts (HH).csv')
df2 = pd.read_csv('2024\Prices & Forecasts (HH).csv')

avp = pd.concat([avp, avp2], ignore_index=True)
dap = pd.concat([dap, dap2], ignore_index=True)
df = pd.concat([df, df2], ignore_index=True)

avp['GMT Time'] = pd.to_datetime(avp['GMT Time'], dayfirst=True)
dap['GMT Time'] = pd.to_datetime(dap['GMT Time'], dayfirst=True)
df['GMT Time'] = pd.to_datetime(df['GMT Time'], dayfirst=True)

frequency = pd.read_csv('2023\Frequency.csv').dropna()
frequency2 = pd.read_csv('2024\Frequency.csv')
excel_base_date = pd.Timestamp('1900-01-01')
frequency2['GMT Time'].iloc[:8725] = excel_base_date + pd.to_timedelta(frequency2['GMT Time'].iloc[:8725].astype(float) - 2, unit='D')
freq = pd.concat([frequency, frequency2], ignore_index=True)
gen = pd.read_csv('2023\Generation by Fuel.csv')
gen2 = pd.read_csv('2024\Generation by Fuel.csv')
gen = pd.concat([gen, gen2], ignore_index=True)
mel = pd.read_csv('2023\MEL below PN.csv').dropna()
mel2 = pd.read_csv('2024\MEL below PN.csv')
mel = pd.concat([mel, mel2], ignore_index=True)

freq['GMT Time'] = pd.to_datetime(freq['GMT Time'], dayfirst=True)
gen['GMT Time'] = pd.to_datetime(gen['GMT Time'], dayfirst=True)
mel['GMT Time'] = pd.to_datetime(mel['GMT Time'], dayfirst=True)

#%%

DCH = avp[['GMT Time', 'Volume Requirements Forecast - DC-H - GB (MW)', 'Ancillary Volume Accepted - DC-H - GB (MW)', 'Ancillary Price - DC-H - GB (£/MW/h)']]
DCL = avp[['GMT Time', 'Volume Requirements Forecast - DC-L - GB (MW)', 'Ancillary Volume Accepted - DC-L - GB (MW)', 'Ancillary Price - DC-L - GB (£/MW/h)']]

DMH = avp[['GMT Time', 'Volume Requirements Forecast - DM-H - GB (MW)', 'Ancillary Volume Accepted - DM-H - GB (MW)', 'Ancillary Price - DM-H - GB (£/MW/h)']]
DML = avp[['GMT Time', 'Volume Requirements Forecast - DM-L - GB (MW)', 'Ancillary Volume Accepted - DM-L - GB (MW)', 'Ancillary Price - DM-L - GB (£/MW/h)']]

DRH = avp[['GMT Time', 'Volume Requirements Forecast - DR-H - GB (MW)', 'Ancillary Volume Accepted - DR-H - GB (MW)', 'Ancillary Price - DR-H - GB (£/MW/h)']]
DRL = avp[['GMT Time', 'Volume Requirements Forecast - DR-L - GB (MW)', 'Ancillary Volume Accepted - DR-L - GB (MW)', 'Ancillary Price - DR-L - GB (£/MW/h)']]

#%%

dap['date'] = dap['GMT Time'].apply(ef.shift_efa).dt.date
dap['EFA Block'] = dap['GMT Time'].dt.hour.apply(ef.get_efa_block)
n2ex = dap.groupby(['date', 'EFA Block'])['Day Ahead Price (N2EX, local) - GB (£/MWh)'].mean().reset_index()
epex = dap.groupby(['date', 'EFA Block'])['Day Ahead Price (EPEX, local) - GB (£/MWh)'].mean().reset_index()

df['date'] = df['GMT Time'].apply(ef.shift_efa).dt.date
df['EFA Block'] = df['GMT Time'].dt.hour.apply(ef.get_efa_block)
ndf = df.groupby(['date', 'EFA Block'])['National Demand Forecast (NDF) - GB (MW)'].mean().reset_index()
dahh = df.groupby(['date', 'EFA Block'])['Day Ahead Price (EPEX half-hourly, local) - GB (£/MWh)'].mean().reset_index()

freq['date'] = freq['GMT Time'].apply(ef.shift_efa).dt.date
freq['EFA Block'] = freq['GMT Time'].dt.hour.apply(ef.get_efa_block)
avg_freq = freq.groupby(['date', 'EFA Block']).mean().reset_index()

gen['date'] = gen['GMT Time'].apply(ef.shift_efa).dt.date
gen['EFA Block'] = gen['GMT Time'].dt.hour.apply(ef.get_efa_block)
avg_gen = gen.groupby(['date', 'EFA Block']).mean().reset_index()

mel['date'] = mel['GMT Time'].apply(ef.shift_efa).dt.date
mel['EFA Block'] = mel['GMT Time'].dt.hour.apply(ef.get_efa_block)
avg_mel = mel.groupby(['date', 'EFA Block']).mean().reset_index()

def merge_dap(datas, target):
    train = target.copy()
    data_type = train.columns[1][31:35]
    train['EFA Block'] = train['GMT Time'].dt.hour.apply(ef.get_efa_block2)
    train['date'] = train['GMT Time'].dt.date
    for dat in datas:
        train = pd.merge(train, dat, on=['date', 'EFA Block'], how='left')
    train['Month'] = train['GMT Time'].dt.month
    train['Day'] = train['GMT Time'].dt.dayofweek + 1
    train = train[['GMT Time', f'Ancillary Price - {data_type} - GB (£/MW/h)', 'Month', 'Day', 'EFA Block', f'Volume Requirements Forecast - {data_type} - GB (MW)', f'Ancillary Volume Accepted - {data_type} - GB (MW)', 'Day Ahead Price (EPEX, local) - GB (£/MWh)','Day Ahead Price (N2EX, local) - GB (£/MWh)', 'National Demand Forecast (NDF) - GB (MW)', 'Day Ahead Price (EPEX half-hourly, local) - GB (£/MWh)']]
    return train

DCH = merge_dap([n2ex, epex, ndf, dahh], DCH)
DCL = merge_dap([n2ex, epex, ndf, dahh], DCL)
DMH = merge_dap([n2ex, epex, ndf, dahh], DMH)
DML = merge_dap([n2ex, epex, ndf, dahh], DML)
DRH = merge_dap([n2ex, epex, ndf, dahh], DRH)
DRL = merge_dap([n2ex, epex, ndf, dahh], DRL)

#%%

def merge_secondarydata(datas, target):
    train = target.copy()
    train['date'] = train['GMT Time'].dt.date
    for dat in datas:
        train = pd.merge(train, dat, on=['date', 'EFA Block'], how='left')
    train = train.drop(columns=['date'])
    return train

DCH = merge_secondarydata([avg_freq, avg_gen, avg_mel], DCH)
DCL = merge_secondarydata([avg_freq, avg_gen, avg_mel], DCL)
DMH = merge_secondarydata([avg_freq, avg_gen, avg_mel], DMH)
DML = merge_secondarydata([avg_freq, avg_gen, avg_mel], DML)
DRH = merge_secondarydata([avg_freq, avg_gen, avg_mel], DRH)
DRL = merge_secondarydata([avg_freq, avg_gen, avg_mel], DRL)

# %%

'''

Adjust for lagged variables 

- DFR Volumes accepted
- DA HH Prices

'''

def adjusted_trainingdata(dat):
    data = dat.copy()
    data_type = data.columns[5][31:35]
    data[f'Ancillary Volume Accepted - {data_type} - GB (MW)'] = data[f'Ancillary Volume Accepted - {data_type} - GB (MW)'].shift(6).fillna(0)
    data[f'Day Ahead Price (EPEX half-hourly, local) - GB (£/MWh)'] = data[f'Day Ahead Price (EPEX half-hourly, local) - GB (£/MWh)'].shift(6).fillna(0)
    columns = data.columns[11:].to_list()
    for column in columns:
        data[column] = data[column].shift(6).fillna(0)
    return data

DCH_train = adjusted_trainingdata(DCH)
DCL_train = adjusted_trainingdata(DCL)
DMH_train = adjusted_trainingdata(DMH)
DML_train = adjusted_trainingdata(DML)
DRH_train = adjusted_trainingdata(DRH)
DRL_train = adjusted_trainingdata(DRL)

# DCH_train.to_csv('John_ML_Data/DCH_train.csv', index=False)
# DCL_train.to_csv('John_ML_Data/DCL_train.csv', index=False)
# DMH_train.to_csv('John_ML_Data/DMH_train.csv', index=False)
# DML_train.to_csv('John_ML_Data/DML_train.csv', index=False)
# DRH_train.to_csv('John_ML_Data/DRH_train.csv', index=False)
# DRL_train.to_csv('John_ML_Data/DRL_train.csv', index=False)


#%%

# testing correlations

def plot_correlation_matrix(data, title):
    plt.figure(figsize=(10, 8))
    corr = data.iloc[:, 1:]
    corr = corr.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title(title)
    plt.show()

plot_correlation_matrix(DCH_train, 'Correlation Matrix for DCH')
plot_correlation_matrix(DCL_train, 'Correlation Matrix for DCL')
plot_correlation_matrix(DMH_train, 'Correlation Matrix for DMH')
plot_correlation_matrix(DML_train, 'Correlation Matrix for DML')
plot_correlation_matrix(DRH_train, 'Correlation Matrix for DRH')
plot_correlation_matrix(DRL_train, 'Correlation Matrix for DRL')
#%%

datas = [DCH_train, DCL_train, DMH_train, DML_train, DRH_train, DRL_train]
# datas = [DCH, DCL, DMH, DML, DRH, DRL]

for dat in datas:
    dat['Total Generation'] = dat.iloc[:, 14:-6].sum(axis=1)
    dat['Residual'] = dat['Total Generation'] - dat['National Demand Forecast (NDF) - GB (MW)']

#%%

datas = [DCH_train, DCL_train, DMH_train, DML_train, DRH_train, DRL_train]
# datas = [DCH, DCL, DMH, DML, DRH, DRL]

df = pd.DataFrame()
for data in datas:
    data_type = data.columns[5][31:35]
    data = data.rename(columns={f'Volume Requirements Forecast - {data_type} - GB (MW)': 'Volume Requirements Forecast', f'Ancillary Volume Accepted - {data_type} - GB (MW)': 'Ancillary Volume Accepted'})
    correlation = data.iloc[:, 1:].corr().iloc[:,:1]
    correlation = correlation[~correlation.isin(correlation.iloc[-8:-2,:])].dropna().iloc[10:]
    # correlation = correlation.iloc[4:-6,:]
    df = pd.concat([df, correlation], axis=1, ignore_index=True)

#%%

df = df.rename(columns={0: 'DCH', 1: 'DCL', 2: 'DMH', 3: 'DML', 4: 'DRH', 5: 'DRL'})
df = df.T
plt.figure(figsize=(12, 10))
sns.heatmap(df, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix for All Datasets')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(np.abs(df), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix for All Datasets (Absolute Values)')
plt.tight_layout()
plt.show()

# %%

datas = [DCH_train, DCL_train, DMH_train.dropna(), DML_train.dropna(), DRH_train.dropna(), DRL_train.dropna()]
# datas = [DCH, DCL, DMH, DML, DRH, DRL]
RMSE_initial = []
RMSES = []

for data in datas:
    df = data.copy()
    data_type = data.columns[5][31:35]
    df = pd.get_dummies(df, columns=['Month', 'Day', 'EFA Block'])
    df = df.drop(columns=['GMT Time'])
    target_col = f'Ancillary Price - {data_type} - GB (£/MW/h)'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    train_size = int(0.8 * len(df))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    preds_initial = model.predict(X_test)
    rmse_initial = np.sqrt(mean_squared_error(y_test, preds_initial))
    print(f"Initial RMSE: {rmse_initial:.3f}")
    RMSE_initial.append(rmse_initial)

    importance = model.feature_importances_
    importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': importance})
    importance_df.sort_values(by='importance', ascending=False, inplace=True)

    threshold = 0.02
    selected_features = importance_df[importance_df['importance'] >= threshold]['feature'].tolist()

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    model_selected = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    model_selected.fit(X_train_selected, y_train)

    preds_selected = model_selected.predict(X_test_selected)
    rmse_selected = np.sqrt(mean_squared_error(y_test, preds_selected))
    print(f"RMSE after feature selection: {rmse_selected:.3f}")
    RMSES.append(rmse_selected)

# %%
