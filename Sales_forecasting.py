import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Sales data 
months = list(range(1, 17))
sales = [125, 125, 130, 125, 135, 140, 135, 140, 150, 155, 140, 160, 165, 150, 175, 170]
data = pd.Series(sales, index=months)

# SES method with alpha 0.3
model = SimpleExpSmoothing(data)
fit = model.fit(smoothing_level=0.3, optimized=False)

# 3 month forecasting (17-19 month)
forecast = fit.forecast(3)
print("Peramalan Bulan 17-19:")
print(forecast)

# actual and forecast data
plt.figure(figsize=(10,5))
plt.plot(data.index, data.values, marker='o', label="Aktual")
plt.plot(range(17, 20), forecast, marker='x', linestyle='--', label="Peramalan")
plt.axvline(x=16.5, color='red', linestyle='--', label="Awal Forecast")
plt.title("Sales Forecasting using SES Method (SES)")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()