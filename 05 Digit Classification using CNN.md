# **Experiment 5: Digit Classification using a Convolutional Neural Network (CNN)**

### **Aim**

To build, train, and evaluate a Convolutional Neural Network (CNN) designed to classify handwritten digits from the MNIST dataset with high accuracy.

### **Algorithm**

1.  **Start**
2.  Import necessary libraries (`tensorflow`, `numpy`, `keras`, `matplotlib`).
3.  Load the MNIST dataset, splitting it into training and testing sets.
4.  Visualize a few sample images from the training set to understand the data.
5.  Pre-process the data:
      * Reshape the input images from (28, 28) to (28, 28, 1) to include a channel dimension for the CNN.
      * Normalize pixel values from the `[0, 255]` range to `[0, 1]`.
      * One-hot encode the integer labels.
6.  Define a `Sequential` CNN model architecture:
      * Add a `Conv2D` layer with 32 filters and `ReLU` activation, followed by a `MaxPooling2D` layer.
      * Add a second `Conv2D` layer with 64 filters and `ReLU`, followed by another `MaxPooling2D` layer.
      * Add a third `Conv2D` layer with 64 filters and `ReLU`.
      * `Flatten` the output from the convolutional layers.
      * Add a `Dense` hidden layer with 64 neurons and `ReLU` activation.
      * Add the final `Dense` output layer with 10 neurons and `softmax` activation.
7.  Compile the model using the `adam` optimizer and `categorical_crossentropy` loss function.
8.  Display the model's architecture using `model.summary()`.
9.  Train the model on the training data, using a portion of it for validation (`validation_split`).
10. Evaluate the final model on the unseen test data to get the test accuracy.
11. Plot the training and validation accuracy over epochs to visualize performance.
12. **Stop**

-----

### **Inputs and Outputs**

  * **Input**: The MNIST dataset, which contains 70,000 28x28 grayscale images of handwritten digits (0-9).
      * 60,000 images for training.
      * 10,000 images for testing.
  * **Outputs**:
    1.  A plot visualizing sample digits from the dataset.
    2.  A summary of the CNN architecture.
    3.  The training progress, showing loss and accuracy for each epoch.
    4.  The final test accuracy of the trained model.
    5.  A plot comparing the training and validation accuracy across epochs.

-----

### **Theory**

#### 1\. MNIST Dataset

The MNIST dataset is a classic collection of 70,000 handwritten digits, commonly used as the "hello, world" of computer vision and deep learning. The goal is to correctly identify the digit (0 through 9) in each 28x28 pixel image.

#### 2\. Convolutional Neural Network (CNN)

A CNN is a specialized type of neural network designed for processing grid-like data, such as images. Unlike a regular Feed-Forward Neural Network, a CNN uses special layers to automatically and adaptively learn spatial hierarchies of features from the input images.

  * **Convolutional Layer (`Conv2D`)**: This is the core building block of a CNN. It applies a set of learnable filters (or kernels) to the input image. Each filter slides over the image to detect specific features like edges, corners, or textures. The output of this operation is called a **feature map**.
  * **Pooling Layer (`MaxPooling2D`)**: This layer is used to down-sample the feature maps, reducing their spatial dimensions. It works by taking the maximum value over a defined window (e.g., 2x2). This helps to make the feature detection more robust to changes in the position of the feature in the image (spatial invariance) and reduces the number of parameters, which helps control overfitting.
  * **Flatten Layer**: This layer converts the final 2D feature maps into a 1D vector, which can then be fed into the standard fully connected (`Dense`) layers for classification.

#### 3\. Data Pre-processing for CNNs

  * **Reshaping**: Standard image datasets are often loaded as `(num_samples, height, width)`. CNNs in Keras expect an additional dimension for the color channels, even for grayscale images. Therefore, we reshape the MNIST data from `(60000, 28, 28)` to `(60000, 28, 28, 1)`.
  * **Normalization and One-Hot Encoding**: These steps are the same as in the previous experiment, ensuring stable training and compatibility with the `categorical_crossentropy` loss function.

-----

### **Code**

```python
# ----------------------------- Import Libraries -----------------------------

from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# ----------------------------- Load and Explore Data -----------------------------

# Load the MNIST dataset (60,000 training and 10,000 test samples)
(xtr, ytr), (xte, yte) = mnist.load_data()

# Print the shapes of the datasets
print("X_train shape:", xtr.shape)
print("y_train shape:", ytr.shape)

# ----------------------------- Visualize Sample Digits -----------------------------
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(xtr[i], cmap='gray')  # Display grayscale image
    plt.title(f"Label: {ytr[i]}")
    plt.xticks([])
    plt.yticks([])
plt.show()

# ----------------------------- Preprocessing -----------------------------

# Reshape input data to fit the CNN input (28x28x1) and normalize
xtr = xtr.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
xte = xte.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

# One-hot encode labels (10 classes: 0–9)
ytr = to_categorical(ytr)
yte = to_categorical(yte)

# ----------------------------- Build CNN Model -----------------------------
model = models.Sequential([
    # First convolutional layer: 32 filters of 3x3
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),  # Reduce spatial size by 2x

    # Second convolutional layer: 64 filters
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Third convolutional layer
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Flatten and connect to dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])

# ----------------------------- Compile the Model -----------------------------
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model architecture
print(model.summary())

# ----------------------------- Train the Model -----------------------------
history = model.fit(xtr, ytr,
                    epochs=15,
                    batch_size=64,
                    validation_split=0.2)  # 20% data used for validation

# ----------------------------- Evaluate on Test Set -----------------------------
test_loss, test_acc = model.evaluate(xte, yte)
print(f'Test accuracy: {test_acc:.4f}')

# ----------------------------- Plot Accuracy -----------------------------
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy (MNIST CNN)')
plt.legend()
plt.grid(True)
plt.show()
```

-----

### **Code Output**

**Sample Digits**
![Digits](./00%20Outputs/digits.png)
**Model Summary**

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 26, 26, 32)          │             320 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 13, 13, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 11, 11, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 5, 5, 64)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 3, 3, 64)            │          36,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 576)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 64)                  │          36,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 10)                  │             650 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 93,322 (364.54 KB)
 Trainable params: 93,322 (364.54 KB)
 Non-trainable params: 0 (0.00 B)
```

**Training Log (abbreviated)**

```
Epoch 1/15
750/750 ━━━━━━━━━━━━━━━━━━━━ 55s 65ms/step - accuracy: 0.8556 - loss: 0.4711 - val_accuracy: 0.9792 - val_loss: 0.0668
Epoch 2/15
750/750 ━━━━━━━━━━━━━━━━━━━━ 48s 64ms/step - accuracy: 0.9818 - loss: 0.0582 - val_accuracy: 0.9875 - val_loss: 0.0459
...
Epoch 14/15
750/750 ━━━━━━━━━━━━━━━━━━━━ 46s 61ms/step - accuracy: 0.9979 - loss: 0.0061 - val_accuracy: 0.9893 - val_loss: 0.0460
Epoch 15/15
750/750 ━━━━━━━━━━━━━━━━━━━━ 47s 63ms/step - accuracy: 0.9983 - loss: 0.0049 - val_accuracy: 0.9907 - val_loss: 0.0541
```

**Final Evaluation**

```
Test accuracy: 0.9910
```

**Accuracy Plot**
![Accuracy Plot](./00%20Outputs/digitClassificationCNN.png)
-----

### **Code Explanation**

1.  **Data Loading & Visualization**: The code first loads the MNIST dataset and uses `matplotlib` to display the first 9 images, confirming the data is loaded correctly.
2.  **Preprocessing**: The training and test images are reshaped to add the channel dimension (1 for grayscale) and normalized to a `[0, 1]` range. The labels are one-hot encoded.
3.  **Model Building**: A `Sequential` model is constructed layer-by-layer. It follows a common CNN pattern: a stack of `Conv2D` and `MaxPooling2D` layers to extract features, followed by a `Flatten` and `Dense` layers for classification.
4.  **Compilation**: `model.compile()` configures the model with the `adam` optimizer and `categorical_crossentropy` loss, which are standard choices for multi-class image classification.
5.  **Training**: `model.fit()` trains the network for 15 epochs. `validation_split=0.2` sets aside 20% of the training data to monitor the model's performance on data it isn't directly training on, which helps in detecting overfitting.
6.  **Evaluation & Plotting**: The model is evaluated on the final, unseen test set to get a definitive measure of its performance. The training history is then used to plot the accuracy curves.

-----

### **Result**

The Convolutional Neural Network trained successfully and achieved a final test accuracy of **99.10%**. The accuracy plot shows that both the training and validation accuracies increased steadily and remained very close to each other, both reaching over 99%.

### **Inference**

The extremely high accuracy demonstrates the effectiveness of CNNs for image classification tasks. By learning spatial features directly from the pixel data, the CNN architecture is far superior to a simple Feed-Forward Neural Network for this type of problem. The close tracking of training and validation accuracy indicates that the model is well-generalized and not significantly overfitting, which is a testament to the robust feature extraction and parameter reduction provided by the convolutional and pooling layers. This experiment successfully showcases a standard and powerful approach to solving image classification problems.