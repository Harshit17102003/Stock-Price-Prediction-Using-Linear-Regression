import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score


stock_symbol = input("Enter the stock_symbol (eg. APPL, NVDA): ").strip().upper()

start_date = input("Enter the start date (YYYY-MM-DD) : ").strip()
end_date = input("Enter the end date (YYYY-MM-DD) : ").strip()

try:
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
except ValueError:
    print("Invalid format of date")
    exit()

print(f"Fetching the data for {stock_symbol}...")
try:
    stock_data = yf.download(stock_symbol, start = start_date, end = end_date)
    if stock_data.empty:
        print(f"No data found for {stock_symbol}. Please try again with the proper symbol.")
    else:
        print(f"Data fetched sucessfully for {stock_symbol}!")
        print(stock_data.head())
except Exception as e:
    print(f"An error has occured {e}")
    exit()

plt.figure(figsize = (10,6))
plt.plot(stock_data['Close'], label = "Closing Price")
plt.title(f"Closing Price of {stock_symbol}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

stock_data["Day"] = np.arange(len(stock_data))
X = stock_data[["Open","Close","Low","High","Volume","Day"]]
y = stock_data["Close"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size =0.2, shuffle = False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(f'\nEvaluation metrix for {stock_symbol} : ')
print(f"Mean squared error is : {mse}")
print(f"R² Score: {r2}")

y_test_flat = y_test.values.ravel() if len(y_test.values.shape) > 1 else y_test
y_pred_flat = y_pred.ravel() if len(y_pred.shape) > 1 else y_pred

predicted_prices = pd.DataFrame({
    'Actual Price' : y_test_flat,
    'Predicted Price' : y_pred_flat})

print("Actual Price V/S Predicted Price")
print(predicted_prices.head(10))

plt.figure(figsize = (10,6))
plt.plot(y_test.values, label = "Actual Price Prices", color = 'blue')
plt.plot(y_pred, label = "Predicted Prices", color = 'orange')
plt.title(f"{stock_symbol} Stock : Actual V\S Predicted Price")
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()