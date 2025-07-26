
# **Experiment 04: Analyzing Optimization and Regularization Techniques**

### **Aim**

To analyze and compare the impact of different weight initialization techniques (**Xavier**, **Kaiming**) and regularization techniques (**Dropout**, **Batch Normalization**) on the performance of a neural network trained on the CIFAR-10 dataset.

### **Algorithm**

1.  **Start**
2.  Import necessary libraries from `tensorflow` and `keras`.
3.  Load the CIFAR-10 dataset and apply pre-processing (normalize pixel values, one-hot encode labels).
4.  Define four distinct `Sequential` models to test each technique:
      * **Model 1**: A baseline model using Xavier (Glorot) weight initialization.
      * **Model 2**: A model using Kaiming (He) weight initialization, suitable for ReLU activations.
      * **Model 3**: A model incorporating a Dropout layer to combat overfitting.
      * **Model 4**: A model incorporating a Batch Normalization layer to stabilize and accelerate training.
5.  Define a common optimizer (`SGD` with momentum) to ensure a fair comparison across all models.
6.  Iterate through each model:
      * Compile the model with the optimizer, `categorical_crossentropy` loss, and `accuracy` metric.
      * Train the model on the training data, using a portion for validation.
      * Evaluate the final model on the unseen test data.
7.  Plot the validation accuracy curves from all four training sessions on a single graph to visually compare their learning dynamics and final performance.
8.  **Stop**

-----

### **Inputs and Outputs**

  * **Input**: The CIFAR-10 dataset.
  * **Outputs**:
    1.  Architecture summaries and final test scores (loss and accuracy) for each of the four models.
    2.  A single comparative plot illustrating the validation accuracy of each model over the training epochs.

-----

### **Theory**

This experiment touches upon some of the most critical concepts for successfully training deep neural networks.

#### 1\. The Challenge: Vanishing and Exploding Gradients

When training a deep network, gradients are calculated via the chain rule during backpropagation. If the weights in the network are too small, the gradients can shrink exponentially as they are propagated backward, becoming so tiny they effectively "vanish." This means the weights of the initial layers never get updated, and the network fails to learn. Conversely, if weights are too large, the gradients can grow exponentially and "explode," causing wildly unstable updates that prevent the model from converging.

**Proper weight initialization is the first line of defense against this problem.**

#### 2\. Weight Initialization Techniques

The goal of smart initialization is to set the initial weights of the network in a way that maintains a stable signal flow (both forward during activation and backward during gradient propagation).

  * **Xavier (or Glorot) Initialization**
    This was one of the first methods to address the issue systematically.

      * **Goal**: To maintain the variance of activations and gradients as they pass through the network. If the variance remains stable (e.g., at 1), the signal is less likely to vanish or explode.
      * **How it works**: It sets the weights by drawing from a distribution (uniform or normal) with a mean of 0 and a carefully chosen variance. The variance is based on the number of input neurons ($n\_{in}$) and output neurons ($n\_{out}$) of the layer:
        $$Var(W) = \frac{2}{n_{in} + n_{out}}$$
      * **Best For**: Xavier initialization was designed assuming a linear or symmetric activation function like `tanh` or `sigmoid`. It works reasonably well with `ReLU` but isn't optimal.

  * **Kaiming (or He) Initialization**
    This method is a direct improvement on Xavier for modern networks that primarily use the ReLU activation function.

      * **The Problem with ReLU**: ReLU sets all negative inputs to 0. This means, on average, it "kills" half of the activations passing through it, which in turn halves the variance of the signal. The Xavier initialization doesn't account for this, so the variance can still decrease through a deep ReLU network.
      * **The Solution**: He initialization modifies the variance formula to compensate for ReLU's behavior. It only considers the number of input neurons:
        $$Var(W) = \frac{2}{n_{in}}$$
      * **Best For**: It is the standard and recommended initialization method for layers that use **ReLU** or its variants (like Leaky ReLU).

#### 3\. Regularization: Fighting Overfitting

**Overfitting** occurs when a model learns the training data too well, capturing not just the underlying patterns but also the noise and random fluctuations. This results in a model with high training accuracy but poor performance on new, unseen data. Regularization techniques are designed to combat this.

  * **Dropout**

      * **How it works**: During each training step, Dropout randomly sets a fraction of neuron activations in a layer to zero. For example, a `Dropout(0.25)` layer will randomly "drop" 25% of its input connections.
      * **Why it works**: This prevents neurons from becoming too co-dependent on each other. A neuron cannot rely on a specific input from another neuron because that input might be dropped at any moment. Therefore, the network is forced to learn more robust and redundant features. It's conceptually similar to training a large ensemble of slightly different networks and averaging their predictions, which is a powerful regularization technique.

  * **Batch Normalization**
    This technique offers both regularization and a significant speed-up in training.

      * **How it works**: It's a layer that normalizes the activations from the previous layer for each mini-batch. It re-centers and re-scales the data to have a **mean of 0 and a standard deviation of 1**. It also has learnable parameters ($\\gamma$ and $\\beta$) that allow the network to scale and shift the normalized output if needed.
      * **Benefits**:
        1.  **Faster Training**: By keeping the distribution of layer inputs stable, it allows for higher, more stable learning rates.
        2.  **Reduces Internal Covariate Shift**: This is the technical term for the problem where the distribution of each layer's inputs changes as the weights of the previous layers are updated. Batch Norm reduces this effect, so layers don't have to constantly re-adapt to a moving target.
        3.  **Regularization**: The normalization is performed on a per-batch basis. The mean and standard deviation of each mini-batch are slightly different from the overall dataset's statistics. This introduces a small amount of noise, which acts as a weak but effective regularizer, sometimes making Dropout unnecessary.

-----

### **Code**

```python
# Import necessary libraries
import tensorflow as tf
import numpy as np
from keras import models, layers, optimizers
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# ----------------------------- Data Loading and Preprocessing -----------------------------
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ----------------------------- Model Definitions -----------------------------

# Model 1: Xavier Initialization
model1 = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(256, activation='relu', kernel_initializer='glorot_uniform'),
    layers.Dense(256, activation='relu', kernel_initializer='glorot_uniform'),
    layers.Dense(10, activation='softmax', kernel_initializer='glorot_uniform')
])

# Model 2: Kaiming (He) Initialization
model2 = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(10, activation='softmax', kernel_initializer='he_normal')
])

# Model 3: With Dropout Layer
model3 = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(256, activation='relu', kernel_initializer='glorot_uniform'),
    layers.Dropout(0.25),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Model 4: With Batch Normalization
model4 = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(10, activation='softmax')
])

# ----------------------------- Training and Evaluation Loop -----------------------------

# Common Optimizer
sgd_optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.9)
models_dict = {'Xavier': model1, 'Kaiming': model2, 'Dropout': model3, 'Batch Norm': model4}
histories = {}

for name, model in models_dict.items():
    print(f"\n--- Training {name} Model ---")
    model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.2, verbose=1)
    histories[name] = history
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"{name} Score - Loss: {score[0]:.4f}, Accuracy: {score[1]:.4f}")

# ----------------------------- Plotting Validation Accuracy -----------------------------
plt.figure(figsize=(12, 8))
for name, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=f'{name} Validation Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Model Validation Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.show()

```

-----

### **Code Output ( abbreviated )**

```
Training shape: (50000, 32, 32, 3)

Model: "sequential_1"
Total params: 855,050 (3.26 MB)
Trainable params: 855,050 (3.26 MB)
Non-trainable params: 0 (0.00 B)

Epoch 1/15 - accuracy: 0.2692 - loss: 1.9886 - val_accuracy: 0.3450 - val_loss: 1.7848
...
Epoch 15/15 - accuracy: 0.5008 - loss: 1.3828 - val_accuracy: 0.4641 - val_loss: 1.5264
313/313 - accuracy: 0.4667 - loss: 1.4948
Xavier Score: [1.5038, 0.4642]

Model: "sequential_2"  (Xavier Init)
Total params: 820,874 (3.13 MB)
Trainable params: 820,874 (3.13 MB)
Non-trainable params: 0 (0.00 B)

Epoch 1/15 - accuracy: 0.2681 - loss: 1.9995 - val_accuracy: 0.3368 - val_loss: 1.7978
...
Epoch 15/15 - accuracy: 0.4891 - loss: 1.4300 - val_accuracy: 0.4380 - val_loss: 1.6010
79/79 - accuracy: 0.4439 - loss: 1.5619
Kaiming Score: [1.5825, 0.4401]

Model: "sequential_3"  (Dropout)
Total params: 820,874 (3.13 MB)
Trainable params: 820,874 (3.13 MB)
Non-trainable params: 0 (0.00 B)

Epoch 1/15 - accuracy: 0.2171 - loss: 2.0999 - val_accuracy: 0.3226 - val_loss: 1.8779
...
Epoch 15/15 - accuracy: 0.4143 - loss: 1.6205 - val_accuracy: 0.4251 - val_loss: 1.6230
79/79 - accuracy: 0.4437 - loss: 1.5931
Dropout Score: [1.5939, 0.4370]

Model: "sequential_4"  (BatchNorm)
Total params: 790,282 (3.01 MB)
Trainable params: 789,770 (3.01 MB)
Non-trainable params: 512 (2.00 KB)

Epoch 1/15 - accuracy: 0.3460 - loss: 1.8692 - val_accuracy: 0.3528 - val_loss: 1.8602
...
Epoch 15/15 - accuracy: 0.5735 - loss: 1.2220 - val_accuracy: 0.4581 - val_loss: 1.5964
79/79 - accuracy: 0.4649 - loss: 1.5611
BatchNorm Score: [1.5674, 0.4624]

```

**Accuracy Plot**

![Accuracy Plot](./00%20Outputs/Optimization%20and%20Weight%20Initialization%20Techniques.png)
-----

### **Code Explanation**

1.  **Data Setup**: The CIFAR-10 dataset is loaded and pre-processed by normalizing the image data and one-hot encoding the labels.
2.  **Model Definitions**: Four distinct `Sequential` models are created to isolate and test each technique. The `kernel_initializer` argument is used for Xavier and Kaiming, while `Dropout` and `BatchNormalization` are added as separate layers. Note the pattern for Batch Norm: it is typically applied *before* the activation function.
3.  **Training Loop**: To ensure a fair comparison, a single `sgd_optimizer` is defined. A dictionary holds the models, and a loop iterates through each one to compile, train, and evaluate it, storing the training history for the final plot.
4.  **Plotting**: `matplotlib` is used to plot the `val_accuracy` from each model's history on the same axes, making it easy to visually compare their performance throughout the training process.

-----

### **Result**

The experiment trained four models, each with a different optimization or regularization strategy. The final test accuracies were:

  * **Xavier**: \~46.4%
  * **Batch Norm**: \~46.2%
  * **Kaiming**: \~44.0%
  * **Dropout**: \~43.7%

The plot shows that the models with **Xavier initialization** and **Batch Normalization** achieved slightly higher and more stable validation accuracy compared to the others in this specific run. The Dropout and Kaiming models performed similarly to each other but lagged slightly behind.

### **Inference**

This experiment highlights several important points:

1.  **Architecture is Key**: All models achieved a modest accuracy of around 45%. This is because the primary limitation is the simple Feed-Forward Network architecture, which is not well-suited for a complex image dataset like CIFAR-10. The benefits of these advanced techniques are often more pronounced in deeper, more complex models (like CNNs).
2.  **Batch Norm and Xavier Show Promise**: Even in this limited architecture, Batch Normalization and Xavier Initialization provided a slight performance edge, suggesting they are robust techniques for stabilizing training.
3.  **Context Matters**: While Kaiming (He) initialization is theoretically superior for ReLU networks, other factors like the specific dataset, optimizer, and network depth can influence the final outcome. Similarly, Dropout's effectiveness can depend on the rate and where it's placed.

In conclusion, this experiment successfully demonstrates *how* to implement these crucial techniques and provides a glimpse into their relative impact. For a more dramatic demonstration of their power, applying them to a deeper Convolutional Neural Network would be the logical next step.