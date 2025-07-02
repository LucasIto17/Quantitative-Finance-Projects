# Quantitative-Finance-Projects

# Apple Stock Forecasting

## Project Overview

This project performs time series forecasting on Apple Inc. (AAPL) stock prices using an ARIMA model. The goal is to predict future adjusted closing prices based on historical data and evaluate model performance.

## Data Source

* **Dataset:** Historical daily stock prices for Apple Inc.
* **File:** `AAPL.csv` (downloaded from Yahoo Finance)
* **Period:** January 1, 2010 to present

## Preprocessing Steps

1. **Loading Data:** Read the CSV file with dates parsed and set as the index.
2. **Sorting:** Ensure the DataFrame is sorted in chronological order.
3. **Filtering Date Range:** Keep records from January 1, 2010 onward.
4. **Cleaning Values:** Remove commas, convert "Adj Close" column to numeric, replace infinite values with NaN, and drop missing entries.

## Methodology

1. **Stationarity Check:** Perform the Augmented Dickey-Fuller (ADF) test to assess stationarity of the series.
2. **ACF & PACF Plots:** Visualize autocorrelation and partial autocorrelation to inform choice of AR and MA terms.
3. **Train-Test Split:** Use an 80/20 split for training and testing sets.
4. **Model Selection:** Grid search over ARIMA(p,1,q) combinations for p and q from 2 to 10. Select the model with the lowest RMSE on the test set.
5. **Forecasting:** Generate forecasts for the test period and compare against actual values.

## Implementation Details

* **Language & Libraries:**

  * Python
  * pandas, numpy
  * matplotlib for plotting
  * statsmodels for ARIMA modeling and stationarity tests
  * scikit-learn for RMSE calculation

* **Key Code:**

  ```python
  # Fit ARIMA and compute RMSE
  for p in range(2, 11):
      for q in range(2, 11):
          model = ARIMA(train['Adj Close'], order=(p, 1, q))
          fit = model.fit()
          forecast = fit.forecast(steps=len(test))
          rmse = mean_squared_error(test['Adj Close'], forecast, squared=False)
  ```

## Results

* The best ARIMA model order and corresponding RMSE are printed to the console.
* A plot shows training data, actual test values, and model forecasts for visual comparison.

## Conclusion

This analysis demonstrates how to apply ARIMA for stock price forecasting. Future work could explore advanced models (e.g., LSTM, Prophet) or feature engineering to incorporate external factors.

## Requirements

* Python 3.7+
* pandas
* numpy
* matplotlib
* statsmodels
* scikit-learn

## Usage

1. Clone this repository.
2. Place `AAPL.csv` in the project directory.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run the notebook or script to reproduce the analysis and forecasts.
