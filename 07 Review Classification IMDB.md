# **Experiment 07: Sentiment Analysis with a Simple Recurrent Neural Network**

### **Aim**

To implement, train, and evaluate a simple Recurrent Neural Network (RNN) for a binary sentiment classification task using the IMDB movie review dataset.

### **Algorithm**

1.  **Start**
2.  Import necessary libraries from `keras` and `tensorflow`.
3.  Set hyperparameters: vocabulary size (`max_features`) and maximum sequence length (`max_words`).
4.  Load the IMDB dataset, pre-tokenized and limited to the specified vocabulary size.
5.  Pre-process the sequences by padding them to a uniform length (`max_words`).
6.  Define a `Sequential` model architecture:
    - An `Embedding` layer to convert integer sequences into dense vector representations.
    - A `SimpleRNN` layer to process the sequences and capture temporal patterns.
    - A final `Dense` output layer with a `sigmoid` activation function for binary classification.
7.  Display the model's architecture using `model.summary()`.
8.  Compile the model using the `adam` optimizer and `binary_crossentropy` loss function.
9.  Train the model on the training data, using a portion for validation.
10. Evaluate the final model's performance on the unseen test data.
11. **Stop**

---

### **Inputs and Outputs**

- **Input**: The IMDB movie review dataset, where each review is a sequence of integers.
- **Outputs**:
  1.  The model's architecture summary.
  2.  The training progress, showing loss and accuracy for each epoch.
  3.  The final evaluation score (loss and accuracy) on the test set.

---

### **Theory**

#### 1\. Sequential Data and Natural Language Processing (NLP)

Unlike standalone images or data points, text is **sequential**. The order of words matters profoundly. "Man bites dog" means something entirely different from "dog bites man." Traditional neural networks (like the FFNNs used previously) process all inputs independently and lack the ability to understand order or context. NLP is a field of AI focused on enabling computers to understand, interpret, and generate human language.

#### 2\. Recurrent Neural Networks (RNNs)

RNNs are a class of neural networks specifically designed to handle sequential data. Their defining feature is a **loop** in their architecture.

- **How it works**: An RNN processes a sequence element by element (e.g., one word at a time). As it processes an element, it combines the input with information from the previous element. This information is carried forward in a **hidden state**, which acts as the network's "memory."
- **The Loop**: At each timestep `t`, the RNN cell takes the current input ($x\_t$) and the previous hidden state ($h\_{t-1}$) to produce the current output and the new hidden state ($h\_t$). This new hidden state $h\_t$ is then passed to the next timestep. This allows the network to maintain a memory of past information to influence future predictions.

#### 3\. The Vanishing Gradient Problem in RNNs

While elegant, the `SimpleRNN` has a major drawback: the **vanishing gradient problem**. During backpropagation, the gradients are passed backward through time. If the network is deep (i.e., the sequences are long), these gradients are repeatedly multiplied by the same weight matrices. If these weights are small, the gradient can shrink exponentially until it becomes effectively zero. This means the network struggles to learn **long-term dependencies**—it might remember the last few words in a sentence but forget the crucial context from the beginning of a long review.

#### 4\. The IMDB Dataset

This is a benchmark dataset for binary sentiment analysis. It contains 50,000 movie reviews, pre-processed by Keras. Each word in the vocabulary is mapped to a unique integer. The dataset is loaded as sequences of these integers. `num_words=5000` restricts the vocabulary to the 5,000 most frequent words to keep the model manageable.

#### 5\. Key Layers for NLP

- **Embedding Layer**:

  - **Purpose**: This is the first layer in any modern NLP model. It converts the integer-based word indices into dense, meaningful vector representations called **word embeddings**.
  - **How it works**: An embedding layer is essentially a lookup table. For each word index in the input sequence, it looks up the corresponding vector. These vectors are initially random but are tuned during training. The network learns to place words with similar meanings close to each other in the vector space. For example, the vectors for "good" and "great" would end up being very similar. `output_dim=32` means each word will be represented by a 32-dimensional vector.

- **Padding Sequences (`pad_sequences`)**:

  - **Purpose**: Neural networks require that all inputs in a batch have the same shape. Since movie reviews have different lengths, we must standardize them.
  - **How it works**: We choose a maximum length (`maxlen`). Any sequence longer than this is truncated. Any sequence shorter than this is "padded" with a special value (usually 0) at the beginning or end until it reaches the desired length.

#### 6\. Binary Cross-Entropy Loss

This is the standard loss function for binary (two-class) classification problems. It measures the distance between the true label (`0` for negative, `1` for positive) and the model's predicted probability (a value between 0 and 1 from the sigmoid function). It penalizes the model more heavily for confident but incorrect predictions.

---

### **Code**

```python
# ----------------------------- Import Libraries -----------------------------
import numpy as np
from keras.datasets import imdb
from keras import layers, models
from keras.preprocessing import sequence

# ----------------------------- Set Hyperparameters -----------------------------
max_features = 5000         # Vocabulary size
max_words = 500             # Max sequence length after padding

# ----------------------------- Load and Preprocess the Dataset -----------------------------
(xtr, ytr), (xte, yte) = imdb.load_data(num_words=max_features)
print(f'{len(xtr)} train sequences\n{len(xte)} test sequences')

# ----------------------------- Pad Sequences -----------------------------
xtr = sequence.pad_sequences(xtr, maxlen=max_words)
xte = sequence.pad_sequences(xte, maxlen=max_words)
print('Train data shape:', xtr.shape)
print('Test data shape:', xte.shape)

# ----------------------------- Build the Model -----------------------------
model = models.Sequential([
    layers.Embedding(max_features, 32, input_length=max_words),
    layers.SimpleRNN(100),
    layers.Dense(1, activation='sigmoid')
])
model.summary()

# ----------------------------- Compile the Model -----------------------------
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ----------------------------- Train the Model -----------------------------
history = model.fit(xtr, ytr,
                    epochs=15,
                    batch_size=64,
                    validation_split=0.2)

# ----------------------------- Evaluate the Model -----------------------------
loss, acc = model.evaluate(xte, yte)
print("Test accuracy:", round(acc*100, 4))

# ----------------------------- Prediction Example -----------------------------
test_seq = np.reshape(xte[7], (1, -1))   
pred = model.predict(test_seq)[0][0]        # Prediction value

if pred >= 0.5:
    print("Positive Review")
else:
    print("Negative Review")

print("Actual Label:", "Positive" if yte[7] == 1 else "Negative")

```

---

### **Code Output**

**Dataset Information**

```
25000 train sequences
25000 test sequences
Train data shape: (25000, 500)
Test data shape: (25000, 500)
```

**Model Summary**

```
Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ embedding_1 (Embedding)         │ (None, 500, 32)        │       160,000 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ simple_rnn_1 (SimpleRNN)        │ (None, 100)            │        13,300 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 1)              │           101 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 173,401 (677.35 KB)
 Trainable params: 173,401 (677.35 KB)
 Non-trainable params: 0 (0.00 B)
```

**Training Log**

```
Epoch 1/15
313/313 ━━━━━━━━━━━━━━━━━━━━ 18s 45ms/step - accuracy: 0.5082 - loss: 0.6961 - val_accuracy: 0.5922 - val_loss: 0.6727
Epoch 2/15
313/313 ━━━━━━━━━━━━━━━━━━━━ 17s 38ms/step - accuracy: 0.6452 - loss: 0.6445 - val_accuracy: 0.6452 - val_loss: 0.6218
Epoch 3/15
313/313 ━━━━━━━━━━━━━━━━━━━━ 12s 40ms/step - accuracy: 0.7368 - loss: 0.5203 - val_accuracy: 0.6912 - val_loss: 0.6227
Epoch 4/15
313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 40ms/step - accuracy: 0.7250 - loss: 0.5373 - val_accuracy: 0.6492 - val_loss: 0.6193
Epoch 5/15
313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 39ms/step - accuracy: 0.7588 - loss: 0.4929 - val_accuracy: 0.7224 - val_loss: 0.5690
Epoch 6/15
313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 39ms/step - accuracy: 0.8077 - loss: 0.4279 - val_accuracy: 0.7466 - val_loss: 0.5833
Epoch 7/15
313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 39ms/step - accuracy: 0.8373 - loss: 0.3777 - val_accuracy: 0.7384 - val_loss: 0.6258
Epoch 8/15
313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 40ms/step - accuracy: 0.8258 - loss: 0.3871 - val_accuracy: 0.7340 - val_loss: 0.6016
Epoch 9/15
313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 39ms/step - accuracy: 0.7309 - loss: 0.5115 - val_accuracy: 0.6084 - val_loss: 0.6772
Epoch 10/15
313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 40ms/step - accuracy: 0.6725 - loss: 0.5813 - val_accuracy: 0.6178 - val_loss: 0.6580
Epoch 11/15
313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 40ms/step - accuracy: 0.7080 - loss: 0.5376 - val_accuracy: 0.6394 - val_loss: 0.6603
Epoch 12/15
313/313 ━━━━━━━━━━━━━━━━━━━━ 12s 38ms/step - accuracy: 0.7427 - loss: 0.4920 - val_accuracy: 0.6650 - val_loss: 0.6818
Epoch 13/15
313/313 ━━━━━━━━━━━━━━━━━━━━ 12s 39ms/step - accuracy: 0.8009 - loss: 0.4227 - val_accuracy: 0.7094 - val_loss: 0.6462
Epoch 14/15
313/313 ━━━━━━━━━━━━━━━━━━━━ 12s 38ms/step - accuracy: 0.8511 - loss: 0.3447 - val_accuracy: 0.7278 - val_loss: 0.6086
Epoch 15/15
313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 39ms/step - accuracy: 0.8797 - loss: 0.3008 - val_accuracy: 0.7238 - val_loss: 0.6231
```

**Model Evaluation**

```
782/782 ━━━━━━━━━━━━━━━━━━━━ 8s 10ms/step - accuracy: 0.7293 - loss: 0.6142
Test accuracy: 73.42
```

**Prediction Example**

```
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 586ms/step
Negative Review
Actual Label: Negative
```

---

### **Code Explanation**

1.  **Data Loading**: `imdb.load_data()` conveniently loads the dataset, already converted into integer sequences. `num_words=max_features` limits the vocabulary size.
2.  **Padding**: `sequence.pad_sequences()` is used to ensure all review sequences have the same length (`maxlen=500`), which is required for batch processing in the RNN.
3.  **Model Building**:
    - `Embedding`: Creates the lookup table to map the 5,000 unique words to 32-dimensional vectors.
    - `SimpleRNN`: This is the core recurrent layer with 100 internal units (neurons) that process the sequence of word embeddings.
    - `Dense`: A final fully connected layer with a single neuron and `sigmoid` activation produces the final probability score (0 to 1) for the review being positive.
4.  **Model Building**: The model is explicitly built using `model.build()` with the specified input shape before showing the summary.
5.  **Compilation**: The model is compiled with `adam` (a common and effective optimizer), `binary_crossentropy` (the standard loss for two-class problems), and `accuracy` as the metric.
6.  **Training**: The model is trained for 15 epochs. `validation_split=0.2` reserves 20% of the training data to monitor performance on data not used for training updates, which helps in identifying overfitting.
7.  **Evaluation**: The model is evaluated on the test set, and the accuracy is displayed as a percentage.
8.  **Prediction Example**: A single test sample is used to demonstrate how the model makes predictions, showing both the predicted sentiment and the actual label.

---

### **Result**

The simple RNN model was successfully trained for 15 epochs and achieved a final test accuracy of approximately **73.42%**. The model demonstrates its ability to learn sentiment patterns from movie reviews, with training accuracy reaching nearly 88% in the final epoch, while validation accuracy peaked around 74% and showed some fluctuation.

### **Inference**

The experiment demonstrates that a simple RNN can learn to perform sentiment analysis, achieving a result significantly better than random chance (50%). However, its performance is modest. The large discrepancy between the high training accuracy and the lower, unstable validation accuracy is a clear sign of **overfitting**. The model began to memorize the training examples rather than learning a generalizable sentiment pattern.

This also highlights the limitations of the `SimpleRNN` layer, namely the vanishing gradient problem, which makes it difficult to capture long-range dependencies in the text. For a more robust solution, more advanced recurrent architectures like **LSTM (Long Short-Term Memory)** or **GRU (Gated Recurrent Unit)**, which are specifically designed to overcome this issue, would be necessary.
