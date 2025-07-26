# **Experiment 03: Feed-Forward Neural Network for CIFAR-10 Classification**

### **Aim**

To implement, train, and evaluate a Feed-Forward Neural Network (FFNN) with three hidden layers for the task of image classification on the CIFAR-10 dataset.

### **Algorithm**

1.  **Start**
2.  Import necessary libraries from `tensorflow` and `keras`.
3.  Load the CIFAR-10 dataset, which is split into training and testing sets.
4.  Pre-process the data:
    - Normalize the pixel values of the images from the `[0, 255]` range to the `[0, 1]` range.
    - Convert the integer labels to one-hot encoded vectors.
5.  Define a `Sequential` model architecture:
    - Start with a `Flatten` layer to convert the 32x32x3 images into a 1D vector.
    - Add three hidden `Dense` layers with `ReLU` activation functions.
    - Add the final output `Dense` layer with 10 neurons (one for each class) and a `softmax` activation function.
6.  Compile the model, specifying the `adam` optimizer, `categorical_crossentropy` loss function, and `accuracy` as the evaluation metric.
7.  Display the model's architecture using `model.summary()`.
8.  Train the model on the training data for a set number of epochs, using the test data for validation.
9.  Evaluate the final model performance on the test set.
10. Plot the training and validation accuracy over epochs to visualize the learning process.
11. **Stop**

---

### **Inputs and Outputs**

- **Input**: The CIFAR-10 dataset, containing 60,000 32x32 color images in 10 classes.
  - 50,000 images for training.
  - 10,000 images for testing.
- **Outputs**:
  1.  A summary of the model's architecture and parameters.
  2.  The training progress, showing loss and accuracy for each epoch.
  3.  The final test accuracy of the trained model.
  4.  A plot comparing the training and validation accuracy across epochs.

---

### **Theory**

#### 1\. CIFAR-10 Dataset

The CIFAR-10 dataset is a widely used benchmark for image classification tasks. It consists of 60,000 32x32 color images across 10 distinct classes: **airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck**.

#### 2\. Feed-Forward Neural Network (FFNN)

An FFNN, or Multi-Layer Perceptron (MLP), is a classic neural network where connections between nodes do not form a cycle. Information moves in only one direction—from the input nodes, through the hidden layers, and to the output nodes.

#### 3\. Data Pre-processing

- **Normalization**: Pixel values in images range from 0 to 255. Dividing them by 255.0 scales these values to a `[0, 1]` range. This helps the optimizer (like Adam) to converge faster and more stably.
- **One-Hot Encoding**: The original labels are integers from 0 to 9. One-hot encoding converts these integers into binary vectors where only one element is "hot" (1) and the rest are "cold" (0). For example, the label `3` becomes `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`. This format is required when using the `categorical_crossentropy` loss function.

#### 4\. Key Layers and Functions

- **Flatten Layer**: This layer converts the multi-dimensional input (e.g., a 32x32x3 image) into a one-dimensional vector (3072 elements). This is necessary to pass image data to a standard `Dense` (fully connected) layer.
- **Dense Layer**: A standard fully connected layer where each neuron is connected to every neuron in the previous layer.
- **ReLU Activation**: The Rectified Linear Unit ($f(x) = \\max(0, x)$) is used in hidden layers to introduce non-linearity, allowing the network to learn complex patterns.
- **Softmax Activation**: This function is used in the output layer for multi-class classification. It converts the raw output scores (logits) from the final layer into a probability distribution over all classes, where the sum of all probabilities is 1.
- **Categorical Cross-Entropy Loss**: This is the standard loss function for multi-class classification problems with one-hot encoded labels. It measures the difference between the predicted probability distribution and the true distribution.

---

### **Code**

```python
import tensorflow as tf
from keras import models, layers
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model
model = models.Sequential()

# Flatten the input for the fully connected layer
model.add(layers.Flatten(input_shape=(32, 32, 3)))

# Three hidden layers with ReLU activation
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))

# Output layer with softmax activation for classification
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=0)
print(f'\nTest loss: {score[0]}')
print(f'Test accuracy: {score[1]}')

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

```

---

### **Code Output**

**Model Summary**

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ flatten (Flatten)                    │ (None, 3072)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 512)                 │       1,573,376 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 256)                 │         131,328 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 128)                 │          32,896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (Dense)                      │ (None, 10)                  │           1,290 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 1,738,890 (6.63 MB)
 Trainable params: 1,738,890 (6.63 MB)
 Non-trainable params: 0 (0.00 B)
```

**Training Log (abbreviated)**

```
Epoch 1/15
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 61s 36ms/step - accuracy: 0.2715 - loss: 2.0145 - val_accuracy: 0.3792 - val_loss: 1.7131
Epoch 2/15
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 65s 41ms/step - accuracy: 0.3852 - loss: 1.7082 - val_accuracy: 0.4066 - val_loss: 1.6424
...
Epoch 14/15
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 67s 42ms/step - accuracy: 0.5430 - loss: 1.2731 - val_accuracy: 0.4955 - val_loss: 1.4581
Epoch 15/15
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 66s 42ms/step - accuracy: 0.5509 - loss: 1.2525 - val_accuracy: 0.4869 - val_loss: 1.4657
```

**Plot**
![FeedForwardNetwork](./00%20Outputs/feedForwardNetwork.png)

---

### **Code Explanation**

1.  **Load Data**: `cifar10.load_data()` downloads and loads the dataset into training and testing splits.
2.  **Pre-process**: The image arrays are divided by `255.0` to normalize them. `to_categorical` transforms the integer labels into one-hot encoded vectors suitable for the loss function.
3.  **Model Definition**: A `Sequential` model is created.
    - `layers.Flatten` unrolls the 3D image arrays into 1D vectors.
    - Three `layers.Dense` are added as hidden layers with progressively fewer neurons (512, 256, 128) and `relu` activation.
    - The final `layers.Dense` output layer has 10 neurons (for 10 classes) and `softmax` activation to generate class probabilities.
4.  **Compilation**: `model.compile()` configures the model for training. It sets the `adam` optimizer, `categorical_crossentropy` loss, and tracks `accuracy`.
5.  **Training**: `model.fit()` trains the network for 15 epochs. It uses `X_train` and `y_train` for training and evaluates performance on `X_test` and `y_test` after each epoch (`validation_data`).
6.  **Plotting**: `matplotlib.pyplot` is used to plot the `accuracy` and `val_accuracy` values stored in the `history` object returned by `model.fit()`.

---

### **Result**

The Feed-Forward Neural Network was successfully trained on the CIFAR-10 dataset for 15 epochs. The final validation accuracy achieved was approximately **48.7%**. The plot of accuracy over epochs shows that the training accuracy consistently increased, while the validation accuracy plateaued around epoch 13 and then slightly decreased.

### **Inference**

The model learned to classify the images significantly better than random chance (which would be 10% accuracy). However, a final accuracy of \~49% is considered low for the CIFAR-10 benchmark.

The key takeaway is visible in the plot: the gap between the rising training accuracy and the flat validation accuracy is a classic sign of **overfitting**. The model started to memorize the training data instead of learning generalizable features. This happens because a simple FFNN does not understand the spatial structure of images (e.g., that pixels close to each other are related).

For better performance on complex image datasets like CIFAR-10, more advanced architectures like **Convolutional Neural Networks (CNNs)** are necessary, as they are specifically designed to process spatial data.
