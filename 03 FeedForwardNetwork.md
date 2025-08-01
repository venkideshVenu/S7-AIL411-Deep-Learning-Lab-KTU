# **Experiment 03: Feed-Forward Neural Network for CIFAR-10 Classification**

### **Aim**

To implement, train, and evaluate a Feed-Forward Neural Network (FFNN) with three hidden layers for the task of image classification on the CIFAR-10 dataset, and to visualize a prediction on a sample image.

### **Algorithm**

1.  **Start**
2.  Import necessary libraries: `tensorflow`, `numpy`, and `matplotlib`.
3.  Load the CIFAR-10 dataset, which is split into training and testing sets (`xtr`, `ytr`, `xte`, `yte`).
4.  Pre-process the data:
      - Normalize the pixel values of the images to the `[0, 1]` range.
      - Convert the integer labels to one-hot encoded vectors.
5.  Define a `Sequential` model architecture:
      - Start with a `Flatten` layer to convert the 32x32x3 images into a 1D vector.
      - Add three hidden `Dense` layers with `ReLU` activation functions.
      - Add the final output `Dense` layer with 10 neurons and a `softmax` activation function.
6.  Compile the model, specifying the `adam` optimizer, `categorical_crossentropy` loss function, and `accuracy` as the evaluation metric.
7.  Train the model on the training data for **5 epochs** with a **batch size of 64**, using the test data for validation.
8.  Evaluate the final model performance on the test set.
9.  Predict the class probabilities for a single sample image from the training set.
10. Visualize the sample image alongside a bar chart of its corresponding prediction probabilities.
11. **Stop**

-----

### **Inputs and Outputs**

  - **Input**: The CIFAR-10 dataset, containing 60,000 32x32 color images in 10 classes.
      - 50,000 images for training.
      - 10,000 images for testing.
  - **Outputs**:
    1.  The training progress, showing loss and accuracy for each epoch.
    2.  The final test accuracy of the trained model, printed as a percentage.
    3.  A plot showing a sample input image and a horizontal bar chart of the model's predicted probabilities for each class.

-----

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

-----

### **Code**

```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Load and Normalize the Data -----------------------------
(xtr, ytr), (xte, yte) = cifar10.load_data()
xtr, xte = xtr / 255.0, xte / 255.0
ytr, yte = to_categorical(ytr), to_categorical(yte)

# ----------------------------- Build the Model -----------------------------
model = Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# ----------------------------- Compile the Model -----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------- Train the Model -----------------------------
history = model.fit(xtr, ytr, epochs=5, batch_size=64, validation_data=(xte, yte))

# ----------------------------- Evaluate the Model -----------------------------
_, acc = model.evaluate(xte, yte, verbose=0)
print("Test accuracy:", round(acc * 100, 4), "%")

# ----------------------------- Predict and Visualize -----------------------------
sample_img = xtr[:1]
pred = model.predict(sample_img)

class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
axs[0].imshow(sample_img[0])
axs[0].axis('off')
axs[0].set_title("Input Image")

axs[1].barh(class_labels, pred[0])
axs[1].set_title("Prediction Probabilities")

plt.tight_layout()
plt.show()

```

-----

### **Code Output**

**Training Log**

```
Epoch 1/5
782/782 ━━━━━━━━━━━━━━━━━━━━ 28s 33ms/step - accuracy: 0.2730 - loss: 1.9997 - val_accuracy: 0.3558 - val_loss: 1.7332
Epoch 2/5
782/782 ━━━━━━━━━━━━━━━━━━━━ 41s 34ms/step - accuracy: 0.3899 - loss: 1.6999 - val_accuracy: 0.3776 - val_loss: 1.7174
Epoch 3/5
782/782 ━━━━━━━━━━━━━━━━━━━━ 39s 32ms/step - accuracy: 0.4302 - loss: 1.5961 - val_accuracy: 0.4256 - val_loss: 1.6138
Epoch 4/5
782/782 ━━━━━━━━━━━━━━━━━━━━ 41s 32ms/step - accuracy: 0.4530 - loss: 1.5317 - val_accuracy: 0.4485 - val_loss: 1.5318
Epoch 5/5
782/782 ━━━━━━━━━━━━━━━━━━━━ 40s 31ms/step - accuracy: 0.4702 - loss: 1.4851 - val_accuracy: 0.4632 - val_loss: 1.5017
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.4615 - loss: 1.5026
Test accuracy: 46.32 %
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 86ms/step
```

**Final Evaluation**

```
Test accuracy: 46.32 %
```

**Plot**
![FeedForwardNetwork](./00%20Outputs/feedForwardNN.png)

-----

### **Code Explanation**

1.  **Load & Pre-process Data**: `cifar10.load_data()` loads the data. The training (`xtr`) and testing (`xte`) image arrays are normalized by dividing by `255.0`. The labels (`ytr`, `yte`) are one-hot encoded using `to_categorical`.
2.  **Model Definition**: A `Sequential` model is defined in a list format.
      - `layers.Flatten` unrolls the 32x32x3 image arrays into 1D vectors of 3072 elements.
      - Three `layers.Dense` are added as hidden layers with `relu` activation.
      - The final `layers.Dense` output layer has 10 neurons and `softmax` activation to generate class probabilities.
3.  **Compilation**: `model.compile()` configures the model with the `adam` optimizer and `categorical_crossentropy` loss function and sets it to monitor `accuracy`.
4.  **Training**: `model.fit()` trains the network for **5 epochs** with a **batch size of 64**. It uses the test set for validation after each epoch.
5.  **Evaluation**: `model.evaluate()` calculates the final loss and accuracy on the unseen test data. The accuracy is then printed.
6.  **Prediction & Visualization**: `model.predict()` is used on a single sample image to get the model's output probabilities. `matplotlib` then creates a figure with two subplots: one showing the input image and the other showing a horizontal bar chart of the predicted probabilities for all 10 classes.

-----

### **Result**

The Feed-Forward Neural Network was successfully trained on the CIFAR-10 dataset for 5 epochs. The final test accuracy achieved was approximately **46.32%**. A sample image was correctly classified, and its prediction probabilities were visualized in a bar chart, demonstrating the model's output on a single instance.

### **Inference**

The model learned to classify images significantly better than random chance (10% accuracy). However, an accuracy of \~46% is still considered low for the CIFAR-10 benchmark. The model's performance is inherently limited because an FFNN treats the image as a flat vector, completely ignoring the crucial **spatial structure** of the pixels (i.e., which pixels are adjacent to each other).

This experiment highlights the limitations of using a simple FFNN for complex image tasks. To achieve higher accuracy, architectures like **Convolutional Neural Networks (CNNs)** are required, as they use convolutional layers specifically designed to preserve and learn from the spatial features in images, such as edges, textures, and shapes.