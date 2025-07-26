# **Experiment 1.3: Implementing y = x with a Neural Network**

### **Aim**

To build and train a minimal neural network using TensorFlow and Keras to learn the simple identity function, $y=x$.

### **Algorithm**

1.  **Start**
2.  Import the necessary Python libraries (`tensorflow`, `numpy`).
3.  Create NumPy arrays `X` and `Y` with a single corresponding element (e.g., `X=[3.0]`, `Y=[3.0]`) to serve as the training data.
4.  Create a `Sequential` model, which is a linear stack of layers.
5.  Add a `Dense` layer to the model with 1 neuron, an input dimension of 1, and no bias term (`use_bias=False`).
6.  Compile the model using Stochastic Gradient Descent (`sgd`) as the optimizer and Mean Squared Error (`mse`) as the loss function.
7.  Train the model for 100 epochs using the training data `X` and `Y`.
8.  Use the trained model to predict the output for a new input value.
9.  Print the predicted output.
10. **Stop**

### **Inputs and Outputs**

  * **Input:**
      * Training Data: A single input-output pair, `x = [3.0]` and `y = [3.0]`.
      * Prediction Input: A new value to test the model, `[12.0]`.
  * **Outputs:**
    1.  The training progress printed to the console, showing the loss value at each epoch.
    2.  The final predicted output for the input `[12.0]`.

### **Theory**

#### 1\. Neural Network

A neural network is a computational model inspired by the structure and function of the human brain. It consists of interconnected nodes called **neurons**, organized in layers. The network learns by adjusting the connections (weights) between these neurons based on the data it is trained on.

#### 2\. Dense Layer

A `Dense` layer is the most basic and common type of layer in a neural network. In a dense layer, every neuron is connected to every neuron in the previous layer. The fundamental operation of a neuron is to calculate a weighted sum of its inputs, add a bias, and then pass the result through an activation function. The equation is:
$$\text{output} = \text{activation}(\text{dot}(\text{inputs}, \text{weights}) + \text{bias})$$
In this specific experiment, our network has a single neuron with one input. The goal is to learn the function $y=x$. The neuron's operation is $y = w \\cdot x + b$. Since we set `use_bias=False`, the bias term $b$ is removed, simplifying the equation to:
$$y = w \cdot x$$
The network's entire task is to learn that the optimal value for the weight $w$ is `1`.

#### 3\. Keras Sequential Model

The `Sequential` model in Keras is a simple way to build a neural network. It allows you to create models layer-by-layer in a linear stack, which is perfect for most common network architectures.

#### 4\. Loss Function: Mean Squared Error (MSE)

As in the previous linear regression experiment, the loss function measures how far the model's prediction is from the actual target value. We use **Mean Squared Error (MSE)**, which calculates the average of the squared differences between the predicted and true values.
$$\text{MSE} = (y_{true} - y_{predicted})^2$$
The model's goal is to adjust its weight $w$ to make this loss as close to zero as possible.

#### 5\. Optimizer: Stochastic Gradient Descent (SGD)

An **optimizer** is an algorithm that modifies the attributes of the neural network, such as its weights, to minimize the loss function. **Stochastic Gradient Descent (SGD)** is a fundamental optimization algorithm. It works by:

1.  Calculating the gradient (slope) of the loss function with respect to the model's weight(s).
2.  Updating the weight(s) by a small amount in the direction opposite to the gradient.
    This iterative process gradually "descends" towards the point of minimum loss, improving the model's accuracy with each step (epoch).

#### 6\. Epoch

An epoch represents one complete pass of the entire training dataset through the neural network. In this experiment, with 100 epochs, the model will look at the training pair `(x=3, y=3)` and update its weight 100 times.

-----

### **Code**

```python
# Import required libraries
import tensorflow as tf                      # TensorFlow for building and training the model
import numpy as np                           # NumPy for numerical operations
from tensorflow import keras
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Dense    # Dense layer for fully connected neural networks

# Define training data
x = np.array([3.0])  # Input data
y = np.array([3.0])  # Output data (target value)

# Create a Sequential model
model = Sequential()

# Add a Dense layer with:
# - 1 neuron
# - no bias term (use_bias=False)
# - input dimension of 1
# - layer named 'D1'
model.add(Dense(1, name='D1', input_dim=1, use_bias=False))

# Compile the model with:
# - Stochastic Gradient Descent (SGD) optimizer
# - Mean Squared Error (MSE) loss function
model.compile(optimizer='sgd', loss='mse')

# Train the model using the input x and target y
# for 100 epochs (iterations over the dataset)
model.fit(x, y, epochs=100)

# Use the trained model to make a prediction for the input value 12
l = model.predict(np.array([12]))

# Print the predicted output
print(l)
```

-----

### **Code Output**

**Training Log (abbreviated)**

```
Epoch 1/100
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 519ms/step - loss: 2.5032
Epoch 2/100
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 156ms/step - loss: 1.6832
...
Epoch 20/100
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 116ms/step - loss: 0.0013
...
Epoch 50/100
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 141ms/step - loss: 9.0042e-09
...
Epoch 75/100
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 151ms/step - loss: 5.1159e-13
...
Epoch 100/100
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - loss: 2.2737e-13
```

**Final Prediction**

```
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 145ms/step
[[11.999998]]
```

-----

### **Code Explanation**

1.  **Import Libraries**: We import `tensorflow` and `numpy`. Key components from `tensorflow.keras` like `Sequential` and `Dense` are also imported.
2.  **Define Data**: `x` and `y` are created as NumPy arrays. We provide a single example (`x=3`, `y=3`) for the model to learn from.
3.  **Create Model**: `model = Sequential()` initializes a new, empty model.
4.  **Add Layer**: `model.add(Dense(...))` adds our single neuron.
      * `1`: Specifies one neuron in the layer.
      * `input_dim=1`: Informs the model to expect a 1-dimensional input.
      * `use_bias=False`: Disables the bias term, forcing the model to learn only the weight `w` in `y = w*x`.
5.  **Compile Model**: `model.compile(...)` configures the model for training.
      * `optimizer='sgd'`: Sets the optimization algorithm to Stochastic Gradient Descent.
      * `loss='mse'`: Sets the loss function to Mean Squared Error.
6.  **Train Model**: `model.fit(x, y, epochs=100)` starts the training process. The model iterates over the data 100 times, adjusting its weight in each epoch to minimize the MSE loss.
7.  **Predict**: `model.predict(np.array([12]))` takes the trained model and uses it to compute the output for a new, unseen input of 12.
8.  **Print Output**: The result of the prediction is printed to the console.

-----

### **Result**

The model was successfully trained for 100 epochs. The training log shows the loss decreasing rapidly, starting at `2.5032` and becoming extremely small (`2.2737e-13`) by the final epoch. When asked to predict the output for an input of `12`, the model returned `[[11.999998]]`.

### **Inference**

This experiment demonstrates that even the simplest neural network can effectively learn a basic mathematical function. By iteratively adjusting its single weight parameter using SGD to minimize the MSE, the model learned that the weight `w` should be extremely close to 1. The final prediction of `11.999998` is almost identical to the correct answer of `12`, proving that the learning process was successful. The decreasing loss at each epoch visualizes this learning process in action.