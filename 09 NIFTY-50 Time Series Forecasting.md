
# **Experiment 9: Time Series Forecasting Prediction for NIFTY-50 Dataset**

### **Aim**

To implement a time series forecasting model using Recurrent Neural Networks (RNNs) to predict NIFTY-50 stock prices based on past historical data.

---

### **Algorithm**

1. **Start**
2. Import the necessary libraries (`pandas`, `numpy`, `keras`, `matplotlib`, `sklearn`).
3. Load the NIFTY-50 dataset and set the `Date` column as the index.
4. Select relevant numerical columns (`Open`, `High`, `Low`, `Price`).
5. Normalize the feature values using `MinMaxScaler` to scale them between 0 and 1.
6. Split the dataset into training (80%) and testing (20%) subsets.
7. Create time-series sequences using `TimeseriesGenerator` to generate input-output pairs.
8. Define a **Sequential** RNN model using LSTM layers:

   * Add an `LSTM` layer with 100 units and `ReLU` activation.
   * Add a `Dense` output layer with 4 neurons (corresponding to the features).
9. Compile the model with the **Adam optimizer** and **mean squared error** loss function.
10. Train the model for 50 epochs using the training generator.
11. Evaluate the model using the test data and generate predictions.
12. Inverse transform the scaled predictions and actual data to get the original price values.
13. Plot:

    * The overall actual vs predicted stock prices.
    * Individual plots for each feature (`Open`, `High`, `Low`, `Turnover`).
14. **Stop**

---

### **Inputs and Outputs**

* **Input:**

  * NIFTY-50 stock dataset containing historical daily values such as `Open`, `High`, `Low`, and `Price`.

* **Outputs:**

  1. Normalized and scaled training/test data.
  2. Model summary showing the LSTM architecture.
  3. Training logs showing loss and accuracy for each epoch.
  4. Predicted vs Actual stock price plots.
  5. Separate visualizations for each feature variable.

---

### **Theory**

#### 1. Time Series Forecasting

Time series forecasting involves predicting future values based on previously observed data points. In financial domains like stock market analysis, this is particularly important for predicting future stock prices, trends, or volatility.

A time series is a sequence of data points indexed in time order. Each observation depends on previous ones, making traditional feed-forward networks unsuitable. Instead, **Recurrent Neural Networks (RNNs)** are used because they can remember temporal dependencies.

#### 2. Recurrent Neural Network (RNN) and LSTM

**RNNs** are specialized neural networks for sequential data. They use loops to retain information from previous inputs to influence the current output.
However, traditional RNNs suffer from **vanishing gradient** problems, which make it difficult to learn long-term dependencies.

To solve this, **Long Short-Term Memory (LSTM)** networks were introduced. They include **memory cells** and **gates** that regulate how much information is remembered or forgotten over time.

**Key Components of LSTM:**

* **Input Gate:** Decides how much new information to store.
* **Forget Gate:** Controls how much old information to discard.
* **Output Gate:** Determines the output of the cell.

This makes LSTMs ideal for financial time series forecasting.

#### 3. Data Preprocessing

1. **Normalization:**
   Scaling input features to a `[0,1]` range using `MinMaxScaler` improves model stability and convergence speed.

2. **Sequence Generation:**
   The `TimeseriesGenerator` from Keras helps in creating sequential data windows (`n_input` time steps) to predict the next step.
   For example, if `n_input=3`, then the model uses 3 consecutive days’ data to predict the next day's prices.

3. **Train-Test Split:**
   Ensures that the model learns on historical data and generalizes on unseen data.

---

### **Code**

```python
# ----------------------------- Import Libraries -----------------------------

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt

# ----------------------------- Load and Inspect Data -----------------------------

# Load dataset with 'Date' as index
data = pd.read_csv('./00 inputs/nifty.csv', index_col='Date', parse_dates=True)

# Select relevant features
new_df = data[['Open', 'High', 'Low', 'Price']]
print(new_df.head())

# ----------------------------- Data Normalization -----------------------------

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(new_df)

# ----------------------------- Train-Test Split -----------------------------

n = int(len(data_scaled) * 0.8)
train_data = data_scaled[:n]
test_data = data_scaled[n:]

# ----------------------------- Time Series Generation -----------------------------

n_input = 3       # number of time steps
n_features = 4    # number of features

generator_train = TimeseriesGenerator(train_data, train_data, length=n_input)
generator_test = TimeseriesGenerator(test_data, test_data, length=n_input)

# ----------------------------- Build RNN Model (LSTM) -----------------------------

model = Sequential()
model.add(LSTM(100, activation='relu'))
model.add(Dense(4))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
print(model.summary())

# ----------------------------- Train the Model -----------------------------

model.fit(generator_train, epochs=50)

# ----------------------------- Evaluate Model -----------------------------

model.evaluate(generator_test)

# Generate Predictions
predictions = model.predict(generator_test)

# Inverse transform predictions and actual data
predictions_original = scaler.inverse_transform(predictions)
test_data_original = scaler.inverse_transform(test_data[n_input:])

# ----------------------------- Plot Actual vs Predicted -----------------------------

plt.figure(figsize=(15, 6))
plt.plot(data.index[n + n_input:], test_data_original, label='Actual Prices', color='blue')
plt.plot(data.index[n + n_input:], predictions_original, label='Predicted Prices', color='orange')
plt.title('NIFTY-50 Stock Price Prediction using RNN')
plt.xlabel('Date')
plt.ylabel('Stock Price (Close)')
plt.legend()
plt.show()

# ----------------------------- Plot Each Feature Separately -----------------------------

variables = ['Open', 'High', 'Low', 'Turnover']

for i, variable in enumerate(variables):
    plt.figure(figsize=(15, 3))
    plt.plot(data.index[n + n_input:], test_data_original[:, i], label=f'Actual {variable}', color='blue')
    plt.plot(data.index[n + n_input:], predictions_original[:, i], label=f'Predicted {variable}', color='red')
    plt.title(f'{variable} Stock Price Prediction using RNN')
    plt.xlabel('Date')
    plt.ylabel(f'{variable} Value')
    plt.legend()
    plt.show()
```

---

### **Code Output**

**Sample Dataset**

```
              Open     High      Low    Price
Date                                        
1995-12-11   882.58   882.58   874.37   877.41
1995-12-18   883.29   883.58   881.17   882.63
1995-12-14   890.81   891.98   884.31   884.34
1995-11-29   845.36   845.36   822.54   845.12
1995-12-19   882.35   884.19   879.54   883.72
```

**Training Log (Abbreviated)**

```
Epoch 1/50
32/32 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.0538 - loss: 0.0468
Epoch 2/50
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.1351 - loss: 0.0142
...
Epoch 50/50
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.4049 - loss: 0.0032
```

**Model Summary**

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ lstm (LSTM)                          │ (None, 100)                 │          42,000 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 4)                   │             404 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
Total params: 42,404 (165.64 KB)
Trainable params: 42,404
Non-trainable params: 0
```

**Evaluation Example**

```
loss: 0.0032 - accuracy: 0.4049
```

**Predicted vs Actual Plot**
![NIFTY Forecast Plot](./00%20Outputs/stock_prediction_using_rnn.png)

**Open Stock Prediction Using RNN**
![Open Stock Prediction](./00%20Outputs/open_stock_prediction.png)

**High Stock Prediction Using RNN**
![High Stock Prediction](./00%20Outputs/high_stock_prediction.png)

**Low Stock Prediction Using RNN**
![Low Stock Prediction](./00%20Outputs/low_stock_prediction.png)

**Turnover Stock Prediction Using RNN**
![Turnover Stock Prediction](./00%20Outputs/turnover_stock_prediction.png)

---

### **Code Explanation**

1. **Data Loading:**
   The dataset is read with `Date` as index, ensuring proper temporal ordering.

2. **Preprocessing:**

   * The selected columns are normalized for efficient training.
   * Data is divided into training (80%) and testing (20%) sets.
   * `TimeseriesGenerator` prepares overlapping sliding windows for RNN input.

3. **Model Design:**

   * A single **LSTM layer** captures sequential dependencies from historical data.
   * The **Dense output layer** predicts four target variables simultaneously.

4. **Training and Evaluation:**

   * Model trained for 50 epochs using MSE loss and Adam optimizer.
   * Evaluated on test data, achieving low loss, indicating good fit.

5. **Visualization:**

   * Plots show both the overall and individual feature predictions versus actual values, confirming the LSTM’s ability to track stock trends.

---

### **Result**

The LSTM-based time series model successfully learned the sequential patterns in NIFTY-50 stock data and predicted future stock price movements effectively.

### **Inference**

The experiment demonstrates that **LSTM-based RNNs** can model complex temporal dependencies in financial data.
While the model achieved moderate accuracy (≈40%) due to stock volatility, the trend predictions are consistent with real-world values.
