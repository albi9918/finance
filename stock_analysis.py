#download data form Yahoo Finance
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

#list of stock symbols to analyze
stocks = ["AAPL", "GOOGL", "META", "NVDA"]

#Download historical stock data for the specified period
data = yf.download(stocks, start="2021-01-01", end="2024-12-31")   
#convert the multi-index into columns for easier manipulation
data = data.stack(level='Ticker').reset_index()
#Rename columns for clarity
data.rename(
    columns={
        'Date': 'Date',
        'Ticker': 'Stock',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    }, 
    inplace=True
)
#Reorder columns
data = data[['Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume']]


# sort data by stock and then date
data = data.sort_values(by=['Stock', 'Date']).reset_index(drop=True)

# extract data for each stock
stock_list = data['Stock'].unique()

#define the comlumns for further analysis
feature_columns = ['Open', 'High', 'Low', 'Volume']
print(data.head())

#simple prepocesor function to fill missing values
def preprocess_data(stock_data):
    return stock_data['Close'].values, stock_data['Date'].values
#simplified AutoReg model
class AutoRegModel:
    def __init__(self, lags, rolling_window, num_forecasts):
        self.lags = lags
        self.rolling_window = rolling_window
        self.num_forecasts = num_forecasts

    def fit(self, y):
        self.y = y

    def predict(self, dates):
        if not self.num_forecasts:
            raise ValueError("Number of forecasts not set.")
        
        #rolling historical predictions
        rolling_preds = [

            AutoReg(self.y[i - self.rolling_window:i], lags=self.lags).fit().forecast(steps=1)[0]
            for i in range(self.rolling_window, len(self.y))

        ]
        hist_dates = dates[self.rolling_window:]
        #future predictions
        future_model = AutoReg(self.y[-self.rolling_window:], lags=self.lags).fit()
        future_preds = future_model.forecast(steps=self.num_forecasts)
        future_dates = pd.date_range(start=pd.to_datetime(dates[-1]) + pd.Timedelta(days=1), periods=self.num_forecasts)
        return np.array(rolling_preds), hist_dates, future_preds, future_dates
    
#process and forecast multiple stocks
def process_stocks(data, stock_list, rolling_window, lags, num_forecasts):
    predictions = {}
    results = []

    for stock in stock_list:
        print(f"Processing stock: {stock}")
        stock_data = data[data['Stock'] == stock]
        y, dates = preprocess_data(stock_data)

        model = AutoRegModel(lags, rolling_window, num_forecasts)
        model.fit(y)
        hist_preds, hist_dates, fut_preds, fut_dates = model.predict(dates)
        
        #calculate RMSE for historical predictions
        true_values = y[rolling_window:]
        rmse = np.sqrt(np.mean((hist_preds - true_values) ** 2))
        results.append([stock, rmse])
        print(f"RMSE for {stock}: {rmse}")

        predictions[stock] = (hist_dates, hist_preds, fut_dates, fut_preds)

    return results, predictions

#plot predictions
def plot_predictions(data, prediction, stock_list):
    fig, axes = plt.subplots(nrows=(len(stock_list) + 1) // 2, ncols=2, figsize=(16, len(stock_list) * 2))
    axes = axes.flatten() if len(stock_list) > 1 else [axes]

    for idx, stock in enumerate(stock_list):
        ax = axes[idx]
        stock_data = data[data['Stock'] == stock]
        stock_data = stock_data[pd.to_datetime(stock_data['Date']) >= pd.Timestamp('2023-06-01')]
        hist_dates, hist_preds, fut_dates, fut_preds = prediction[stock]
        hist_mask = pd.to_datetime(hist_dates) >= pd.Timestamp('2023-06-01')
        hist_dates, hist_preds = hist_dates[hist_mask], hist_preds[hist_mask]
        ax.plot(stock_data['Date'], stock_data['Close'], label='Historical Prices', color='blue')
        ax.plot(hist_dates, hist_preds, label='Historical Forecasts', color='green', alpha =0.75)
        ax.plot(fut_dates, fut_preds, label='Future Forecasts', color='red', linestyle='--')
        ax.set_title(stock)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
    plt.savefig('stock_forecasts.png')
    plt.tight_layout()
    plt.show()

results, predictions = process_stocks(data, stock_list, rolling_window=120, lags=15, num_forecasts=10)
plot_predictions(data, predictions, stock_list)