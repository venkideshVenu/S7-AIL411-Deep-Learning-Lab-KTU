# Experiment 08: Analyze and Visualize the Performance Change Using LSTM and GRU Instead of Simple RNN

## Aim

To compare and analyze the performance of **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** models for sentiment classification on the IMDB movie review dataset, and visualize how their performance changes across epochs.

---

## Algorithm

1. **Load the IMDB dataset** with the top 10,000 most frequent words.
2. **Preprocess the data** by padding/truncating reviews to a fixed length (200 words).
3. **Build the LSTM model**:

   * Embedding layer
   * LSTM layer (128 units)
   * Dense output layer with sigmoid activation
4. **Train the LSTM model** with binary cross-entropy loss and Adam optimizer.
5. **Build the GRU model** with the same architecture but replace LSTM with GRU.
6. **Train the GRU model** under identical conditions.
7. **Evaluate both models** on the test set.
8. **Plot training/validation accuracy and loss** for both models.
9. **Compare performance metrics** (accuracy, loss) to analyze differences.

---

## Inputs and Outputs

* **Input**:

  * IMDB dataset reviews (preprocessed as integer sequences).
  * Each sequence padded/truncated to 200 tokens.

* **Output**:

  * Model accuracy and loss values on training, validation, and test sets.
  * Comparison plots of accuracy and loss for LSTM vs GRU.

---

## Theory

### Dataset

* **IMDB Movie Reviews** dataset: Binary sentiment classification (positive/negative).
* Contains **25,000 training** and **25,000 testing reviews**.
* Reviews are preprocessed as integer-encoded word indices.

### RNN vs LSTM vs GRU

* **Simple RNNs** often struggle with long-term dependencies due to the vanishing gradient problem.
* **LSTM (Long Short-Term Memory)** introduces gates (input, forget, output) and a cell state to better capture long-term dependencies.
* **GRU (Gated Recurrent Unit)** simplifies LSTM by combining forget and input gates into a single update gate, making it computationally lighter but often competitive with LSTMs.

### Loss Function

* **Binary Crossentropy**:

  $$
  L = -\frac{1}{N}\sum (y \cdot \log(\hat{y}) + (1-y)\cdot \log(1-\hat{y}))
  $$

### Optimizer

* **Adam Optimizer** is used for adaptive learning rate optimization.

---

## Code

```python
# Experiment 08: Analyze and Visualize the Performance of LSTM vs GRU

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GRU
import matplotlib.pyplot as plt

# ----------------------------- Hyperparameters -----------------------------
num_words = 10000      # Use the top 10,000 most frequent words
max_length = 200       # Each review is padded/truncated to 200 words

# ----------------------------- Load & Preprocess Data -----------------------------
(xtr, ytr), (xte, yte) = imdb.load_data(num_words=num_words)

# Pad sequences to fixed length
xtr, xte = pad_sequences(xtr, maxlen=max_length), pad_sequences(xte, maxlen=max_length)

# ----------------------------- LSTM Model -----------------------------
l_model = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=max_length),  # Word embedding
    LSTM(128),                                                                # LSTM layer
    Dense(1, activation='sigmoid')                                            # Output layer for binary classification
])

# Compile LSTM model
l_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train LSTM model
l_history = l_model.fit(xtr, ytr, validation_split=0.2, epochs=5, batch_size=64)

# Evaluate on test set
loss, acc = l_model.evaluate(xte, yte)
print("LSTM Test accuracy:", round(acc*100, 4))

# ----------------------------- GRU Model -----------------------------
g_model = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=max_length),  # Word embedding
    GRU(128),                                                                 # GRU layer
    Dense(1, activation='sigmoid')                                            # Output layer
])

# Compile GRU model
g_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train GRU model
g_history = g_model.fit(xtr, ytr, validation_split=0.2, epochs=5, batch_size=64)

# Evaluate on test set
loss, acc = g_model.evaluate(xte, yte)
print("GRU Test accuracy:", round(acc*100, 4))

# ----------------------------- Plot Accuracy & Loss -----------------------------
plt.figure(figsize=(12,6))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(l_history.history['accuracy'], label='LSTM (train)')
plt.plot(l_history.history['val_accuracy'], label='LSTM (validation)')
plt.plot(g_history.history['accuracy'], label='GRU (train)')
plt.plot(g_history.history['val_accuracy'], label='GRU (validation)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Training vs Validation Accuracy")
plt.legend()

# Loss plot
plt.subplot(1,2,2)
plt.plot(l_history.history['loss'], label='LSTM (train)')
plt.plot(l_history.history['val_loss'], label='LSTM (validation)')
plt.plot(g_history.history['loss'], label='GRU (train)')
plt.plot(g_history.history['val_loss'], label='GRU (validation)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Training vs Validation Loss")
plt.legend()

plt.show()
```

---

## Code Output (Sample Logs)

### LSTM Training

```
Epoch 1/5 - accuracy: 0.7100 - val_accuracy: 0.8630
Epoch 2/5 - accuracy: 0.8960 - val_accuracy: 0.8622
Epoch 3/5 - accuracy: 0.9315 - val_accuracy: 0.8262
Epoch 4/5 - accuracy: 0.9500 - val_accuracy: 0.8548
Epoch 5/5 - accuracy: 0.9668 - val_accuracy: 0.8542
Test accuracy: 84.92%
```

### GRU Training

```
Epoch 1/5 - accuracy: 0.6772 - val_accuracy: 0.8618
Epoch 2/5 - accuracy: 0.8992 - val_accuracy: 0.8748
Epoch 3/5 - accuracy: 0.9407 - val_accuracy: 0.8568
Epoch 4/5 - accuracy: 0.9617 - val_accuracy: 0.8654
Epoch 5/5 - accuracy: 0.9797 - val_accuracy: 0.8560
Test accuracy: 85.15%
```

### Plots

* **Accuracy plot**: Shows training vs validation accuracy for LSTM and GRU.
* **Loss plot**: Shows training vs validation loss for LSTM and GRU.

![Plot](./00%20Outputs/LSTMvsGRU.png)

---

## Code Explanation

* **Data Preprocessing**: The IMDB reviews are converted into sequences of integers and padded to uniform length.
* **Model Construction**:

  * Both models use an **Embedding layer** to map words into dense vectors.
  * One uses **LSTM**, the other **GRU** for sequence modeling.
  * A **Dense output layer** with sigmoid activation predicts sentiment (positive/negative).
* **Training**: Each model is trained for 5 epochs with a validation split of 20%.
* **Evaluation**: Accuracy is measured on the test dataset.
* **Visualization**: Matplotlib plots show accuracy and loss trends for comparison.

---

## Result

* **LSTM Test Accuracy**: \~84.92%
* **GRU Test Accuracy**: \~85.15%

Both models performed well, with GRU slightly outperforming LSTM in this setup.

---

## Inference

* Both **LSTM and GRU** models effectively handled sentiment classification on IMDB data.
* **GRU achieved slightly better test accuracy** while being computationally simpler, making it a good alternative to LSTM in many cases.
* Validation accuracy fluctuated, indicating possible overfitting beyond 3 epochs.
* **Insight**: GRU can match or surpass LSTM performance with fewer parameters, making it more efficient for deployment.
