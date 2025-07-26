# **Experiment 9: Digit Classification using Transfer Learning with VGG19**

### **Aim**

To implement and analyze a transfer learning approach for digit classification by using a pre-trained VGG19 network on the MNIST dataset. The goal is to observe the performance and efficiency gains from leveraging pre-learned features.

### **Algorithm**

1.  **Start**
2.  Import necessary libraries including `tensorflow`, `numpy`, and `keras`.
3.  Load the MNIST dataset.
4.  Pre-process the images to be compatible with the VGG19 model:
      * Resize the 28x28 grayscale images to 32x32.
      * Convert the single-channel grayscale images into a three-channel format by repeating the channel.
      * Normalize pixel values to a `[0, 1]` range.
5.  One-hot encode the integer labels.
6.  Load the VGG19 model, pre-trained on the ImageNet dataset. Exclude the final classification layer (`include_top=False`) and set the input shape to (32, 32, 3).
7.  **Freeze** the layers of the pre-trained VGG19 base model to prevent their weights from being updated during training.
8.  Build a new `Sequential` model by stacking the frozen VGG19 base, a `Flatten` layer, and new `Dense` layers for classification.
9.  Compile the new model using the `adam` optimizer and `categorical_crossentropy` loss.
10. Train the model on the prepared MNIST data for a few epochs.
11. Display the model summary, highlighting the number of trainable vs. non-trainable parameters.
12. Evaluate the model's final performance on the test set.
13. Plot the training and validation accuracy to visualize the learning process.
14. **Stop**

-----

### **Inputs and Outputs**

  * **Input**: The MNIST dataset of 28x28 grayscale handwritten digits.
  * **Outputs**:
    1.  The model summary detailing the architecture.
    2.  The final test accuracy score.
    3.  A plot of training and validation accuracy over the epochs.

-----

### **Theory**

#### 1\. Transfer Learning

**Transfer learning** is a machine learning technique where a model developed for a task is reused as the starting point for a model on a second, related task. Instead of building a model from scratch, you leverage the "knowledge" (i.e., learned features, weights, and biases) from a model that has already been trained on a large and general dataset like ImageNet. This approach is highly effective because it saves significant training time and often leads to better performance, especially when the target dataset is small.

#### 2\. VGG19

**VGG19** is a very deep Convolutional Neural Network (CNN) architecture with 19 layers (16 convolutional and 3 fully connected). It is known for its simple and uniform structure, which consists of stacked 3x3 convolution layers followed by max-pooling layers. By being trained on the massive ImageNet dataset (over 14 million images across 1000 classes), VGG19 has learned a rich hierarchy of visual features, from simple edges and colors to complex textures and object parts.

#### 3\. Feature Extraction

In this experiment, we use VGG19 for **feature extraction**. The process involves:

1.  **Using the Convolutional Base**: We take the pre-trained VGG19 model but chop off its original top classification layers. The remaining convolutional base acts as a powerful, fixed feature extractor.
2.  **Freezing Weights**: We freeze the weights of the convolutional base (`base_model.trainable = False`). This is crucial because it prevents the valuable, pre-learned features from being corrupted or lost during training on our new, smaller dataset.
3.  **Adding a New Classifier**: We stack our own new `Dense` layers (the classifier head) on top of the frozen base.
4.  **Training Only the New Classifier**: When we train the model, only the weights of our new `Dense` layers are updated. The model learns to classify handwritten digits by using the sophisticated features provided by the frozen VGG19 base.

#### 4\. Adapting MNIST for VGG19

The VGG19 model has specific input requirements. To use it with MNIST, we must adapt our data:

  * **Input Size**: VGG19's architecture requires a minimum input size of 32x32 pixels. Therefore, we must resize the 28x28 MNIST images.
  * **Color Channels**: VGG19 was trained on ImageNet's color images, which have 3 channels (RGB). MNIST images are grayscale with only 1 channel. We must convert our images to a 3-channel format, which is typically done by simply repeating the single grayscale channel three times.

-----

### **Code**

```python
# ----------------------------- Import Libraries -----------------------------
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.applications import VGG19
from keras import layers, models
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# ----------------------------- Load and Explore Dataset -----------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f'Original X_train shape: {X_train.shape}')

# ----------------------------- Resize and Preprocess Images -----------------------------
# Resize to 32x32 and convert grayscale (1 channel) to RGB (3 channels)
X_train = np.repeat(tf.image.resize(X_train[..., np.newaxis], (32, 32)).numpy(), 3, axis=-1)
X_test = np.repeat(tf.image.resize(X_test[..., np.newaxis], (32, 32)).numpy(), 3, axis=-1)
print(f'Processed X_train shape: {X_train.shape}')

# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# ----------------------------- One-Hot Encode Labels -----------------------------
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# ----------------------------- Load Pre-trained VGG19 Model -----------------------------
base_model = VGG19(
    include_top=False,      # Exclude final classification layers
    weights='imagenet',       # Use pre-trained ImageNet weights
    input_shape=(32, 32, 3) # Input shape compatible with resized MNIST
)
base_model.trainable = False  # Freeze base model layers

# ----------------------------- Build the Full Model -----------------------------
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')  # Output layer for 10 digit classes
])

# ----------------------------- Compile the Model -----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------- Train the Model -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# ----------------------------- Model Summary and Evaluation -----------------------------
model.summary()
score = model.evaluate(X_test, y_test)
print(f'\nTest Accuracy: {score[1]:.4f}')

# ----------------------------- Plot Training History -----------------------------
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('VGG19 Transfer Learning on MNIST')
plt.legend()
plt.grid(True)
plt.show()

```

-----

### **Code Output**

**Training Log (abbreviated)**

```
Epoch 1/5
750/750 ━━━━━━━━━━━━━━━━━━━━ 1211s 2s/step - accuracy: 0.8209 - loss: 0.6391 - val_accuracy: 0.9491 - val_loss: 0.1759
Epoch 2/5
750/750 ━━━━━━━━━━━━━━━━━━━━ 1324s 2s/step - accuracy: 0.9526 - loss: 0.1529 - val_accuracy: 0.9542 - val_loss: 0.1417
Epoch 3/5
750/750 ━━━━━━━━━━━━━━━━━━━━ 1247s 2s/step - accuracy: 0.9622 - loss: 0.1200 - val_accuracy: 0.9629 - val_loss: 0.1143
Epoch 4/5
750/750 ━━━━━━━━━━━━━━━━━━━━ 1142s 2s/step - accuracy: 0.9656 - loss: 0.1051 - val_accuracy: 0.9691 - val_loss: 0.0963
Epoch 5/5
750/750 ━━━━━━━━━━━━━━━━━━━━ 1183s 2s/step - accuracy: 0.9685 - loss: 0.0980 - val_accuracy: 0.9675 - val_loss: 0.0983
```

**Model Summary**

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ vgg19 (Functional)                   │ (None, 1, 1, 512)           │      20,024,384 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 512)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 256)                 │         131,328 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 10)                  │           2,570 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 20,426,080 (77.92 MB)
 Trainable params: 133,898 (523.04 KB)
 Non-trainable params: 20,024,384 (76.39 MB)
```

**Final Evaluation**

```
Test Accuracy: 0.9705
```

**Accuracy Plot**

![Accuracy Plot](./00%20Outputs/digitClassificationVGGnet-19.png)
-----

### **Code Explanation**

1.  **Data Pre-processing**: The code first resizes the 28x28 images to 32x32. `X_train[..., np.newaxis]` adds a channel dimension, which is then repeated three times using `np.repeat` to simulate an RGB image. The data is then normalized and labels are one-hot encoded.
2.  **Load Pre-trained Base**: `VGG19()` is called with `include_top=False` to get only the convolutional feature-extraction layers. `weights='imagenet'` ensures it's loaded with the pre-trained weights.
3.  **Freeze Base Model**: `base_model.trainable = False` is a critical step. It freezes all 20 million parameters of the VGG19 base, ensuring that only the new classifier's weights will be updated.
4.  **Build Full Model**: A new `Sequential` model is created. The frozen `base_model` is added as the first layer, followed by a `Flatten` layer and the new classification `Dense` layers.
5.  **Training**: The model is trained for only 5 epochs. Since the feature extraction is already done, the model only needs to learn the relatively simple task of mapping these features to the 10 digit classes.
6.  **Summary and Evaluation**: The model summary clearly shows that out of \~20.4 million total parameters, only \~134,000 are trainable. This demonstrates the efficiency of the transfer learning approach.

-----

### **Result**

The model using VGG19 for transfer learning achieved a very high test accuracy of **97.05%** after just 5 epochs of training. The plot shows that the validation accuracy rapidly increased to over 94% in the first epoch alone and continued to improve steadily, closely tracking the training accuracy.

### **Inference**

This experiment powerfully demonstrates the effectiveness of transfer learning.

  * **High Performance, Fast**: By leveraging the rich visual features learned by VGG19 on ImageNet, the model was able to achieve an excellent result on a completely different dataset (MNIST) with very little training. It did not need to learn basic features like edges, curves, and textures from scratch.
  * **Parameter Efficiency**: Despite VGG19 being a massive model, we only had to train \~134,000 parameters. This makes the training process computationally much cheaper and faster than training a large model from the ground up.

Compared to building a custom CNN from scratch (as in a previous experiment), which required more epochs to reach its peak performance, transfer learning provides a powerful and efficient shortcut to achieving high accuracy, especially when computational resources or the size of the target dataset are limited.