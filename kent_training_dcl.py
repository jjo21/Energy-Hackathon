#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kent_functions as kf
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, cross_val_predict
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor, XGBClassifier

avp = pd.read_csv('2023\Ancillary Volumes & Prices (4H).csv')
dap = pd.read_csv('2023\Day-Ahead Price (1H).csv').dropna()
df1 = pd.read_csv('2023\Prices & Forecasts (HH).csv')
drift = pd.read_csv('2023\drift.csv')

avp2 = pd.read_csv('2024\Ancillary Volumes & Prices (4H).csv')
dap2 = pd.read_csv('2024\Day-Ahead Price (1H).csv')
df2 = pd.read_csv('2024\Prices & Forecasts (HH).csv')

tavp = pd.concat([avp, avp2], ignore_index=True)
tavp['GMT Time'] = pd.to_datetime(tavp['GMT Time'], dayfirst=True)

tdap = pd.concat([dap, dap2], ignore_index=True)
tdap['GMT Time'] = pd.to_datetime(tdap['GMT Time'], dayfirst=True)

paf = pd.concat([df1, df2], ignore_index=True)
paf['GMT Time'] = pd.to_datetime(paf['GMT Time'], dayfirst=True)

drift['GMT Time'] = pd.to_datetime(drift['GMT Time'])
drift = drift.drop(columns=['Unnamed: 0'])

aggregated_ndf = kf.aggregate_all(paf, 'National Demand Forecast (NDF) - GB (MW)')
aggregated_EPEX = kf.aggregate_all(tdap, 'Day Ahead Price (EPEX, local) - GB (£/MWh)')

for i in range(1, len(aggregated_ndf.columns)):
    colname = aggregated_ndf.columns[i]
    aggregated_ndf = aggregated_ndf.rename(columns={f'{colname}':f'{colname} NDF'})

for i in range(1, len(aggregated_EPEX.columns)):
    colname = aggregated_EPEX.columns[i]
    aggregated_EPEX = aggregated_EPEX.rename(columns={f'{colname}':f'{colname} EPEX'})

dch = tavp.copy()
dch = dch.drop(dch.columns[1:], axis=1)
dch = dch.merge(aggregated_EPEX, how='left', on='GMT Time')
#dch = dch.merge(aggregated_ndf, how='left', on='GMT Time')
#dch = dch.merge(drift, how='left', on='GMT Time')
dch = kf.time_filters(dch)
#dch = dch[dch['Post EAC'] == 1]
#for i in range(1, 8):
#    dch[f'Day Lag {i}'] = tavp['Ancillary Price - DC-H - GB (£/MW/h)'].shift(i*6).copy()
dch['DC-L Price'] = tavp['Ancillary Price - DC-L - GB (£/MW/h)'].copy()
dch['Trend'] = np.where(dch['DC-L Price'] - dch['DC-L Price'].shift(1) > 0, 1, 0)

target1 = 'Trend'
features = dch.columns[1:-2].to_list()
dch = dch.dropna(subset=features + [target1])
X = dch[features]
y = dch[target1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#model = make_pipeline(StandardScaler(), LinearRegression())
#model = make_pipeline(StandardScaler(), Lasso(alpha=0.05))
model1 = XGBClassifier(n_estimators=500, learning_rate=0.05)

model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Trend RMSE: {rmse:.2f}')
print(f'R²: {r2:.2f}')
#print(y_test)

plt.figure(figsize=(10, 4))
plt.bar(range(len(model1.feature_importances_)), model1.feature_importances_)
plt.title(f'DC-L Trend Feature Importances RMSE: {rmse:.2f}')
plt.savefig('kent_result_plots/dcl_trend_feature_importances.png')

target2 = 'DC-L Price'
y = dch[target2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model2 = XGBRegressor(n_estimators=500, learning_rate=0.05)
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

plt.figure(figsize=(10, 4))
plt.bar(range(len(model2.feature_importances_)), model2.feature_importances_)
plt.title('DC-L Price Feature Importances')
plt.savefig('kent_result_plots/dcl_price_feature_importances.png')

plt.figure(figsize=(10, 4))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title(f'Actual vs Predicted DC-L Prices RMSE: {rmse:.2f} R²: {r2:.2f}')
plt.savefig('kent_result_plots/dcl_price_predictions.png')