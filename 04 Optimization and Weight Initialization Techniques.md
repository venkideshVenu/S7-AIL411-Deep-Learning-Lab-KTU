# **Experiment 04: Analyzing Optimization and Regularization Techniques**

### **Aim**

To analyze and compare the impact of different weight initialization techniques (**Xavier**, **Kaiming**) and various regularization techniques (**Dropout**, **Batch Normalization**, **L1**, **L2**, and **L1+L2 combined**) on the performance of neural networks trained on the CIFAR-10 dataset.

### **Algorithm**

1.  **Start**
2.  Import necessary libraries from `tensorflow`, `keras`, and `matplotlib`.
3.  Load the CIFAR-10 dataset and apply pre-processing (normalize pixel values to [0,1], one-hot encode labels).
4.  Define seven distinct `Sequential` models to test each technique:
    - **Model 1**: Xavier (Glorot) weight initialization with tanh activation.
    - **Model 2**: Kaiming (He) weight initialization, optimized for ReLU activations.
    - **Model 3**: Dropout regularization layer to combat overfitting.
    - **Model 4**: Batch Normalization layer to stabilize and accelerate training.
    - **Model 5**: L1 regularization (Lasso) with deep architecture for sparsity.
    - **Model 6**: L2 regularization (Ridge) with deep architecture for weight decay.
    - **Model 7**: Combined L1+L2 regularization (Elastic Net) for hybrid approach.
5.  Create a unified training function to ensure fair comparison across all models with customizable optimizers and batch sizes.
6.  Iterate through each model:
    - Compile the model with SGD optimizer, `categorical_crossentropy` loss, and `accuracy` metric.
    - Train the model on the training data for 15 epochs, using 20% for validation.
    - Evaluate the final model on the unseen test data and record scores.
7.  Plot the validation accuracy curves from all seven training sessions on a single graph to visually compare their learning dynamics and final performance.
8.  **Stop**

---

### **Inputs and Outputs**

- **Input**: The CIFAR-10 dataset.
- **Outputs**:
  1.  Architecture summaries and final test scores (loss and accuracy) for each of the seven models.
  2.  A comprehensive comparative plot illustrating the validation accuracy of each model over the training epochs.
  3.  Detailed analysis of different optimization and regularization strategies.

---

### **Theory**

This experiment touches upon some of the most critical concepts for successfully training deep neural networks.

#### 1\. The Challenge: Vanishing and Exploding Gradients

When training a deep network, gradients are calculated via the chain rule during backpropagation. If the weights in the network are too small, the gradients can shrink exponentially as they are propagated backward, becoming so tiny they effectively "vanish." This means the weights of the initial layers never get updated, and the network fails to learn. Conversely, if weights are too large, the gradients can grow exponentially and "explode," causing wildly unstable updates that prevent the model from converging.

**Proper weight initialization is the first line of defense against this problem.**

#### 2\. Weight Initialization Techniques

The goal of smart initialization is to set the initial weights of the network in a way that maintains a stable signal flow (both forward during activation and backward during gradient propagation).

- **Xavier (or Glorot) Initialization**
  This was one of the first methods to address the issue systematically.

  - **Goal**: To maintain the variance of activations and gradients as they pass through the network. If the variance remains stable (e.g., at 1), the signal is less likely to vanish or explode.
  - **How it works**: It sets the weights by drawing from a distribution (uniform or normal) with a mean of 0 and a carefully chosen variance. The variance is based on the number of input neurons ($n\_{in}$) and output neurons ($n\_{out}$) of the layer:
    $$Var(W) = \frac{2}{n_{in} + n_{out}}$$
  - **Best For**: Xavier initialization was designed assuming a linear or symmetric activation function like `tanh` or `sigmoid`. It works reasonably well with `ReLU` but isn't optimal.

- **Kaiming (or He) Initialization**
  This method is a direct improvement on Xavier for modern networks that primarily use the ReLU activation function.

  - **The Problem with ReLU**: ReLU sets all negative inputs to 0. This means, on average, it "kills" half of the activations passing through it, which in turn halves the variance of the signal. The Xavier initialization doesn't account for this, so the variance can still decrease through a deep ReLU network.
  - **The Solution**: He initialization modifies the variance formula to compensate for ReLU's behavior. It only considers the number of input neurons:
    $$Var(W) = \frac{2}{n_{in}}$$
  - **Best For**: It is the standard and recommended initialization method for layers that use **ReLU** or its variants (like Leaky ReLU).

#### 3\. Regularization: Fighting Overfitting

**Overfitting** occurs when a model learns the training data too well, capturing not just the underlying patterns but also the noise and random fluctuations. This results in a model with high training accuracy but poor performance on new, unseen data. Regularization techniques are designed to combat this.

- **Dropout**

  - **How it works**: During each training step, Dropout randomly sets a fraction of neuron activations in a layer to zero. For example, a `Dropout(0.25)` layer will randomly "drop" 25% of its input connections.
  - **Why it works**: This prevents neurons from becoming too co-dependent on each other. A neuron cannot rely on a specific input from another neuron because that input might be dropped at any moment. Therefore, the network is forced to learn more robust and redundant features. It's conceptually similar to training a large ensemble of slightly different networks and averaging their predictions, which is a powerful regularization technique.

- **Batch Normalization**
  This technique offers both regularization and a significant speed-up in training.

  - **How it works**: It's a layer that normalizes the activations from the previous layer for each mini-batch. It re-centers and re-scales the data to have a **mean of 0 and a standard deviation of 1**. It also has learnable parameters ($\\gamma$ and $\\beta$) that allow the network to scale and shift the normalized output if needed.
  - **Benefits**:
    1.  **Faster Training**: By keeping the distribution of layer inputs stable, it allows for higher, more stable learning rates.
    2.  **Reduces Internal Covariate Shift**: This is the technical term for the problem where the distribution of each layer's inputs changes as the weights of the previous layers are updated. Batch Norm reduces this effect, so layers don't have to constantly re-adapt to a moving target.
    3.  **Regularization**: The normalization is performed on a per-batch basis. The mean and standard deviation of each mini-batch are slightly different from the overall dataset's statistics. This introduces a small amount of noise, which acts as a weak but effective regularizer, sometimes making Dropout unnecessary.

---

### **Code**

```python
# ----------------------------------- Imports -----------------------------------
from keras import layers, models, optimizers, regularizers
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# ----------------------------------- Data Loading -----------------------------------
(xtr, ytr), (xte, yte) = cifar10.load_data()

# Normalize pixel values [0,1]
xtr = xtr.astype('float32') / 255.0
xte = xte.astype('float32') / 255.0

# One-hot encode labels
ytr = to_categorical(ytr, 10)
yte = to_categorical(yte, 10)

print("Training shape:", xtr.shape)

# ----------------------------------- Models -----------------------------------

# Model 1: Xavier Initialization
model1 = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(512, activation='tanh', kernel_initializer='glorot_normal'),
    layers.Dense(256, activation='tanh', kernel_initializer='glorot_normal'),
    layers.Dense(10, activation='softmax', kernel_initializer='glorot_normal')
])

# Model 2: Kaiming Initialization
model2 = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(10, activation='softmax', kernel_initializer='he_normal')
])

# Model 3: Dropout Regularization
model3 = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(256, activation='relu', kernel_initializer='glorot_uniform'),
    layers.Dropout(0.25),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Model 4: Batch Normalization
model4 = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])

# Model 5: L1 Regularization
model5 = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(512, activation='relu', kernel_initializer='glorot_normal',
                 kernel_regularizer=regularizers.l1(0.0001)),
    layers.Dense(256, activation='relu', kernel_initializer='glorot_normal',
                 kernel_regularizer=regularizers.l1(0.0001)),
    layers.Dense(128, activation='relu', kernel_initializer='glorot_normal',
                 kernel_regularizer=regularizers.l1(0.0001)),
    layers.Dense(64, activation='relu', kernel_initializer='glorot_normal',
                 kernel_regularizer=regularizers.l1(0.0001)),
    layers.Dense(32, activation='relu', kernel_initializer='glorot_normal',
                 kernel_regularizer=regularizers.l1(0.0001)),
    layers.Dense(10, activation='softmax')
])

# Model 6: L2 Regularization
model6 = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(512, activation='relu', kernel_initializer='glorot_normal',
                 kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dense(256, activation='relu', kernel_initializer='glorot_normal',
                 kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dense(128, activation='relu', kernel_initializer='glorot_normal',
                 kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dense(64, activation='relu', kernel_initializer='glorot_normal',
                 kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dense(32, activation='relu', kernel_initializer='glorot_normal',
                 kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dense(10, activation='softmax')
])

# Model 7: L1 + L2 Regularization
model7 = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(512, activation='relu', kernel_initializer='glorot_normal',
                 kernel_regularizer=regularizers.l1_l2(l1=0.00005, l2=0.00005)),
    layers.Dense(256, activation='relu', kernel_initializer='glorot_normal',
                 kernel_regularizer=regularizers.l1_l2(l1=0.00005, l2=0.00005)),
    layers.Dense(128, activation='relu', kernel_initializer='glorot_normal',
                 kernel_regularizer=regularizers.l1_l2(l1=0.00005, l2=0.00005)),
    layers.Dense(64, activation='relu', kernel_initializer='glorot_normal',
                 kernel_regularizer=regularizers.l1_l2(l1=0.00005, l2=0.00005)),
    layers.Dense(32, activation='relu', kernel_initializer='glorot_normal',
                 kernel_regularizer=regularizers.l1_l2(l1=0.00005, l2=0.00005)),
    layers.Dense(10, activation='softmax')
])

# ----------------------------------- Training Function -----------------------------------
def compile_and_train(model, optimizer, name, batch_size=32, epochs=15):
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(xtr, ytr, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    score = model.evaluate(xte, yte, batch_size=batch_size)
    print(f"{name} Score:", score)
    return history

# ----------------------------------- Optimizers -----------------------------------
histories = {}
histories['Xavier'] = compile_and_train(model1, optimizers.SGD(learning_rate=0.01, momentum=0.9), "Xavier", 32)
histories['Kaiming'] = compile_and_train(model2, optimizers.SGD(learning_rate=0.01, momentum=0.7), "Kaiming", 32)
histories['Dropout'] = compile_and_train(model3, optimizers.SGD(learning_rate=0.01, momentum=0.9), "Dropout", 32)
histories['BatchNorm'] = compile_and_train(model4, optimizers.SGD(learning_rate=0.01, momentum=0.9), "BatchNorm", 128)
histories['L1'] = compile_and_train(model5, optimizers.SGD(learning_rate=0.01, momentum=0.9, clipnorm=1.0), "L1", 128)
histories['L2'] = compile_and_train(model6, optimizers.SGD(learning_rate=0.01, momentum=0.9), "L2", 128)
histories['L1L2'] = compile_and_train(model7, optimizers.SGD(learning_rate=0.01, momentum=0.9), "L1L2", 128)

# ----------------------------------- Plotting -----------------------------------
plt.figure(figsize=(10,6))
for name, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=name)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Model Validation Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.show()

```

---

### **Code Output**

**Dataset Information**

```
Training shape: (50000, 32, 32, 3)
```

**Xavier Initialization Model**

```
/usr/local/lib/python3.12/dist-packages/keras/src/layers/reshaping/flatten.py:37: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ flatten (Flatten)               │ (None, 3072)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 512)            │     1,573,376 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 256)            │       131,328 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 10)             │         2,570 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 1,707,274 (6.51 MB)
 Trainable params: 1,707,274 (6.51 MB)
 Non-trainable params: 0 (0.00 B)

Xavier Score: [1.5470373630523682, 0.46480000019073486]
```

**Kaiming Initialization Model**

```
Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ flatten_1 (Flatten)             │ (None, 3072)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 256)            │       786,688 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_4 (Dense)                 │ (None, 128)            │        32,896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_5 (Dense)                 │ (None, 10)             │         1,290 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 820,874 (3.13 MB)
 Trainable params: 820,874 (3.13 MB)
 Non-trainable params: 0 (0.00 B)

Kaiming Score: [1.4844555854797363, 0.486299991607666]
```

**Dropout Regularization Model**

```
Model: "sequential_2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ flatten_2 (Flatten)             │ (None, 3072)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_6 (Dense)                 │ (None, 256)            │       786,688 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_7 (Dense)                 │ (None, 128)            │        32,896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_8 (Dense)                 │ (None, 10)             │         1,290 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 820,874 (3.13 MB)
 Trainable params: 820,874 (3.13 MB)
 Non-trainable params: 0 (0.00 B)

Dropout Score: [1.6123390197753906, 0.4278999865055084]
```

**Batch Normalization Model**

```
Model: "sequential_3"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ flatten_3 (Flatten)             │ (None, 3072)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_9 (Dense)                 │ (None, 256)            │       786,688 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 256)            │         1,024 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation (Activation)         │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_10 (Dense)                │ (None, 10)             │         2,570 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 790,282 (3.01 MB)
 Trainable params: 789,770 (3.01 MB)
 Non-trainable params: 512 (2.00 KB)

BatchNorm Score: [1.5781753063201904, 0.4487000107765198]
```

**L1 Regularization Model**

```
Model: "sequential_4"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ flatten_4 (Flatten)             │ (None, 3072)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_11 (Dense)                │ (None, 512)            │     1,573,376 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_12 (Dense)                │ (None, 256)            │       131,328 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_13 (Dense)                │ (None, 128)            │        32,896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_14 (Dense)                │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_15 (Dense)                │ (None, 32)             │         2,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_16 (Dense)                │ (None, 10)             │           330 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 1,748,266 (6.67 MB)
 Trainable params: 1,748,266 (6.67 MB)
 Non-trainable params: 0 (0.00 B)

L1 Score: [3.5293359756469727, 0.4896000027656555]
```

**L2 Regularization Model**

```
Model: "sequential_5"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ flatten_5 (Flatten)             │ (None, 3072)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_17 (Dense)                │ (None, 512)            │     1,573,376 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_18 (Dense)                │ (None, 256)            │       131,328 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_19 (Dense)                │ (None, 128)            │        32,896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_20 (Dense)                │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_21 (Dense)                │ (None, 32)             │         2,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_22 (Dense)                │ (None, 10)             │           330 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 1,748,266 (6.67 MB)
 Trainable params: 1,748,266 (6.67 MB)
 Non-trainable params: 0 (0.00 B)

L2 Score: [1.551052212715149, 0.5037999749183655]
```

**L1 + L2 Regularization Model**

```
Model: "sequential_6"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ flatten_6 (Flatten)             │ (None, 3072)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_23 (Dense)                │ (None, 512)            │     1,573,376 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_24 (Dense)                │ (None, 256)            │       131,328 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_25 (Dense)                │ (None, 128)            │        32,896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_26 (Dense)                │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_27 (Dense)                │ (None, 32)             │         2,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_28 (Dense)                │ (None, 10)             │           330 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 1,748,266 (6.67 MB)
 Trainable params: 1,748,266 (6.67 MB)
 Non-trainable params: 0 (0.00 B)

L1L2 Score: [2.145805597305298, 0.49950000643730164]
```

**Validation Accuracy Comparison Plot**

![Optimization and Weight Initialization Techniques Comparison](./00%20Outputs/Optimization%20and%20Weight%20Initialization%20Techniques.png)

---


### **Code Explanation**

1.  **Data Setup**: The CIFAR-10 dataset is loaded and pre-processed by normalizing the image data to [0,1] range and one-hot encoding the labels for multi-class classification.

2.  **Model Definitions**: Seven distinct `Sequential` models are created to test different optimization and regularization techniques:
    - **Xavier (Glorot) Initialization**: Uses `glorot_normal` initialization with `tanh` activation
    - **Kaiming (He) Initialization**: Uses `he_normal` initialization optimized for ReLU activations
    - **Dropout Regularization**: Incorporates a 25% dropout layer to prevent overfitting
    - **Batch Normalization**: Applies batch normalization to stabilize training
    - **L1 Regularization**: Adds L1 penalty (0.0001) to encourage sparsity
    - **L2 Regularization**: Adds L2 penalty (0.0001) for weight decay
    - **L1+L2 Regularization**: Combines both L1 and L2 penalties for hybrid regularization

3.  **Training Function**: A unified `compile_and_train` function ensures fair comparison across all models, with customizable optimizers, batch sizes, and training epochs.

4.  **Training Process**: Each model is trained for 15 epochs with different batch sizes optimized for their specific technique. SGD optimizer with momentum is used consistently.

5.  **Evaluation and Plotting**: Final test scores are recorded and validation accuracy curves are plotted for visual comparison of learning dynamics.

---

### **Result**

The experiment trained seven models, each with a different optimization or regularization strategy. The final test accuracies were:

- **L2 Regularization**: ~50.38% (Best Performance)
- **L1+L2 Combined**: ~49.95%
- **L1 Regularization**: ~48.96%
- **Kaiming (He) Init**: ~48.63%
- **Xavier (Glorot) Init**: ~46.48%
- **Batch Normalization**: ~44.87%
- **Dropout**: ~42.79%

The results show that **L2 regularization** achieved the highest test accuracy, followed by the combined L1+L2 approach. Interestingly, the deeper architectures used for regularization models (5-6 layers) with proper weight penalties outperformed the simpler initialization-focused models.

### **Inference**

This comprehensive experiment reveals several important insights:

1. **Regularization Techniques Excel**: L2 and combined L1+L2 regularization significantly outperformed other methods, demonstrating their effectiveness in controlling overfitting even in simple feed-forward networks.

2. **Architecture Depth Matters**: The regularization models used deeper architectures (512→256→128→64→32→10) compared to initialization models, contributing to better feature learning capacity.

3. **Kaiming vs Xavier**: Kaiming initialization showed marginal improvement over Xavier for ReLU-based networks, confirming theoretical predictions about ReLU-optimized weight initialization.

4. **Batch Normalization Limitations**: While batch normalization accelerated training, it didn't translate to superior final accuracy in this simple architecture, possibly due to the relatively shallow network depth.

5. **Dropout Trade-offs**: Dropout showed the lowest performance, likely due to the already limited capacity of the simple network being further constrained by random neuron deactivation.

6. **Training Dynamics**: The validation accuracy curves revealed that regularization techniques (L1, L2, L1+L2) provided more stable and consistent improvement throughout training epochs.

This experiment successfully demonstrates the practical implementation and comparative analysis of crucial deep learning optimization techniques. The results emphasize that regularization strategies can be particularly effective for preventing overfitting and improving generalization, even in relatively simple network architectures. For more complex datasets and deeper networks, these techniques become even more critical for achieving optimal performance.
```
