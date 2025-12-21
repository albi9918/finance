#download data form Yahoo Finance
import yfinance as yf
import pandas as pd
from utilsforecast.plotting import plot_series
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import *
from statsforecast import StatsForecast
from statsforecast.models import Naive, HistoricAverage, WindowAverage, SeasonalNaive
import os

meta = yf.Ticker("MSFT")
google = yf.Ticker("GOOGL")
apple = yf.Ticker("AAPL")
nvidia = yf.Ticker("NVDA")
print(apple.info)
print(meta.info)
print(google.info)
print(nvidia.info)

#creating DataFrame of the most important element

data_finance = {
    "Company": ["META", "GOOGLE", "APPLE", "NVIDIA"],
    "Sector": [
        meta.info['sector'],
        google.info['sector'],
        apple.info['sector'],
        nvidia.info['sector']
    ],
    "P/E Ratio": [
        meta.info['trailingPE'],
        google.info['trailingPE'],
        apple.info['trailingPE'],
        nvidia.info['trailingPE']
    ],
    "Beta": [
        meta.info['beta'],
        google.info['beta'],
        apple.info['beta'],
        nvidia.info['beta']
    ]
}

df_finance = pd.DataFrame(data_finance)
df_finance = df_finance.sort_values("P/E Ratio", ascending=False).reset_index(drop=True)
print(df_finance)

#describing info of the database

for key, value in meta.info.items():
    print(f"{key}: {value}")

print("-"*100)

for key, value in google.info.items():
    print(f"{key}: {value}")

print("-"*100)

for key, value in apple.info.items():
    print(f"{key}: {value}")

print("-"*100)

for key, value in nvidia.info.items():
    print(f"{key}: {value}")


#evaluating history values for the four companies

start_date = "2024-01-01"
end_date = "2024-12-31"
data = meta.history(start=start_date, end=end_date)
print(data.to_string())

print("-"*150)

start_date = "2024-01-01"
end_date = "2024-12-31"
data2 = google.history(start=start_date, end=end_date)
print(data2.to_string())

print("-"*150)

start_date = "2024-01-01"
end_date = "2024-12-31"
data3 = apple.history(start=start_date, end=end_date)
print(data3.to_string())

print("-"*150)

start_date = "2024-01-01"
end_date = "2024-12-31"
data4 = nvidia.history(start=start_date, end=end_date)
print(data4.to_string())

# Grafici disabilitati - solo valori numerici
# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(1, 4, figsize=(15, 4))
# data['Close'].plot(ax=axs[0], title="Meta Stock")
# data2['Close'].plot(ax=axs[1], title="Google Stock")
# data3['Close'].plot(ax=axs[2], title="Apple Stock")
# data4['Close'].plot(ax=axs[3], title="Nvidia Stock")
# for ax in axs:
#     ax.set_ylabel("Close Price (USD)")
#     ax.set_xlabel("Data")
# plt.tight_layout()
# plt.show()

# plt.plot(figsize=(15, 4))
# data['Close'].plot(label="Meta Stock")
# data2['Close'].plot(label="Google Stock")
# data3['Close'].plot(label="Apple Stock")
# data4['Close'].plot(label="Nvidia Stock")
# plt.ylabel("Close Price (USD)")
# plt.xlabel("Data")
# plt.legend(loc="upper left")
# plt.tight_layout()
# plt.show()

#DataFrame with first open, last close, earnings in the year and earnings(%)

companies = {
    "META": data,
    "GOOGLE": data2,
    "APPLE": data3,
    "NVIDIA": data4
}

rows = []

for name,df in companies.items():
  first_open = df["Open"].iloc[0]
  last_close = df["Close"].iloc[-1]
  earnings = last_close - first_open
  earningspct = (earnings / first_open) * 100
  rows.append({
      "Company": name,
      "First Open": first_open,
      "Last Close": last_close,
      "Earnings": earnings,
      "Earnings(%)": earningspct
  })


df = pd.DataFrame(rows)
df = df.sort_values("Earnings(%)", ascending=False).reset_index(drop=True)
print(df)

# Preparazione dati per forecasting
# StatsForecast richiede colonne: unique_id, ds, y

def prepare_data_for_forecast(data, company_name):
    """Prepara i dati nel formato richiesto da StatsForecast"""
    df = data[['Close']].reset_index()
    df.columns = ['ds', 'y']
    df['unique_id'] = company_name
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)  # Rimuovi timezone
    return df[['unique_id', 'ds', 'y']]

# Crea DataFrame unificato per tutte le aziende
df_meta_fc = prepare_data_for_forecast(data, 'META')
df_google_fc = prepare_data_for_forecast(data2, 'GOOGLE')
df_apple_fc = prepare_data_for_forecast(data3, 'APPLE')
df_nvidia_fc = prepare_data_for_forecast(data4, 'NVIDIA')

# Combina tutti i dati
df_all = pd.concat([df_meta_fc, df_google_fc, df_apple_fc, df_nvidia_fc], ignore_index=True)
print("\nDati preparati per forecasting:")
print(df_all.head(10))

# Definizione modelli statistici
horizon = 30  # Previsione per 30 giorni

models = [
    Naive(),
    HistoricAverage(),
    WindowAverage(window_size=12),
    SeasonalNaive(season_length=5)  # 5 giorni lavorativi = 1 settimana
]

# Creazione e training del modello
sf = StatsForecast(
    models=models,
    freq='B',  # Business day frequency
    n_jobs=1  # Usa un solo processo per evitare problemi su macOS
)

# Fit del modello sui dati
sf.fit(df_all)

# Genera previsioni
forecasts = sf.predict(h=horizon)
print("\nPrevisioni generate:")
print(forecasts)

# Stampa previsioni dettagliate (solo numeri, senza grafici)
companies_fc = ['META', 'GOOGLE', 'APPLE', 'NVIDIA']
print("\n" + "="*80)
print("PREVISIONI DETTAGLIATE PER AZIENDA")
print("="*80)
for company in companies_fc:
    print(f"\n{company}:")
    print(forecasts[forecasts['unique_id'] == company].to_string())

# Plot dei forecast per ogni azienda (grafici separati)
import matplotlib.pyplot as plt

for company in companies_fc:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Dati storici dell'azienda
    company_data = df_all[df_all['unique_id'] == company]
    company_preds = forecasts[forecasts['unique_id'] == company]
    
    ax.plot(company_data['ds'], company_data['y'], label='Historical Data', marker='o', linewidth=2, markersize=2)
    ax.plot(company_preds['ds'], company_preds['Naive'], label='Naive', marker='_', linestyle='--')
    ax.plot(company_preds['ds'], company_preds['HistoricAverage'], label='Historic Average', marker='s', linestyle='--', markersize=3)
    ax.plot(company_preds['ds'], company_preds['WindowAverage'], label='Window Average', marker='.', linestyle='--')
    ax.plot(company_preds['ds'], company_preds['SeasonalNaive'], label='Seasonal Naive', marker='d', linestyle='--', markersize=3)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price (USD)')
    ax.set_title(f'Forecast for {company}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    # plt.savefig(f'forecast_{company.lower()}.png', dpi=150)
    # print(f"Grafico salvato come 'forecast_{company.lower()}.png'")
    plt.show()

# Valutazione deli modelli base
test = df_all.groupby('unique_id').tail(horizon)
train = df_all.drop(test.index).reset_index(drop=True)
sf.fit(df=train)
preds = sf.predict(h=horizon)
eval_df = pd.merge(test, preds, 'left', ['ds', 'unique_id'])
evaluation = evaluate(
    eval_df, 
    metrics=[mae],
    )
print("\nValutazione dei modelli:")
print(evaluation.head())

evaluation = evaluation.drop(['unique_id'], axis=1).groupby('metric').mean().reset_index()
print("\nValutazione media dei modelli:")
print(evaluation.head())

# grafico della valutazione dei modelli
methods = evaluation.columns[1:].tolist()
values = evaluation.iloc[0, 1:].tolist()

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, values)

for bar, vlaue in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{vlaue:.3f}", ha='center', va='bottom', fontweight='bold')
plt.xlabel('Methods')
plt.ylabel('MAE')
plt.tight_layout()
plt.show()    

# autoARIMA
from statsforecast.models import AutoARIMA
unique_ids = ["APPLE", "META", "GOOGLE", "NVIDIA"]
small_train = train[train['unique_id'].isin(unique_ids)]
small_test = test[test['unique_id'].isin(unique_ids)]

models = [
    AutoARIMA(seasonal=False, alias="ARIMA"),
    AutoARIMA(season_length=5, alias="SARIMA")
]

sf = StatsForecast(models=models, freq='D')
sf.fit(df=small_train)
arima_preds = sf.predict(h=horizon)  # Reset index per avere unique_id come colonna

# Rimuovi colonna index se presente
if 'index' in arima_preds.columns:
    arima_preds = arima_preds.drop('index', axis=1)

print("\nPrevisioni ARIMA:")
print(arima_preds.head())
print(arima_preds.columns.tolist())

arima_eval_df = pd.merge(arima_preds, eval_df, 'inner', ['ds', 'unique_id'])
arima_eval = evaluate(
    arima_eval_df, 
    metrics=[mae],
    )
print("\nValutazione dei modelli AutoARIMA:")
print(arima_eval.head())

arima_eval = arima_eval.drop(['unique_id'], axis=1, errors='ignore').groupby('metric').mean().reset_index()
print("\nValutazione media dei modelli AutoARIMA:")
print(arima_eval.head())

# Plot ARIMA predictions con plot_series
fig = plot_series(
    df=df_all, 
    forecasts_df=arima_preds, 
    ids=["APPLE", "META", "GOOGLE", "NVIDIA"], 
    max_insample_length=60
)
fig.tight_layout()
fig.savefig('arima_forecast.png', dpi=150)
print("\nGrafico salvato come 'arima_forecast.png'")
plt.show()
