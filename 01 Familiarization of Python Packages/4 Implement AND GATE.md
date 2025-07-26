
# **Experiment 1.4: Implementing a Logical AND Gate**

### **Aim**

To build, train, and test a simple neural network using TensorFlow and Keras that learns and replicates the behavior of a logical AND gate.

### **Algorithm**

1.  **Start**
2.  Import the necessary Python libraries (`tensorflow`, `numpy`, `keras`).
3.  Create NumPy arrays `X` and `Y` representing the input-output truth table for the AND gate.
4.  Initialize a `Sequential` model.
5.  Add two `Dense` layers to the model: a hidden layer with a ReLU activation function and an output layer with a Sigmoid activation function.
6.  Compile the model using the `Adam` optimizer and `mse` (Mean Squared Error) as the loss function.
7.  Train the model for 350 epochs using the input `X` and target output `Y`.
8.  Predict the output for all four possible input combinations using the trained model.
9.  Print the predicted outputs.
10. **Stop**

### **Inputs and Outputs**

  * **Input:** The complete truth table for a 2-input AND gate.
      * `X = [[0, 0], [0, 1], [1, 0], [1, 1]]`
      * `Y = [[0], [0], [0], [1]]`
  * **Outputs:**
    1.  The training progress (loss per epoch) printed to the console.
    2.  The model's predicted output for each of the four input pairs.

### **Theory**

#### 1\. Logical AND Gate

A logical AND gate is a fundamental digital logic gate that implements logical conjunction. It takes two or more binary inputs and produces a single binary output. The output is `1` (True) if and only if **all** of its inputs are `1`. Otherwise, the output is `0` (False).

| Input A | Input B | Output |
| :---: | :---: | :----: |
|   0   |   0   |   0    |
|   0   |   1   |   0    |
|   1   |   0   |   0    |
|   1   |   1   |   1    |

#### 2\. Activation Functions

Activation functions are crucial components of a neural network that introduce non-linearity into the model. Without them, a neural network, no matter how many layers it has, would behave just like a single-layer linear model.

  * **ReLU (Rectified Linear Unit):** This is the most commonly used activation function in hidden layers. It is defined as:
    $$f(x) = \max(0, x)$$
    It outputs the input directly if it is positive, and zero otherwise. This simple function helps the network learn complex patterns without suffering from the vanishing gradient problem that affects other functions like Sigmoid in deep networks.

  * **Sigmoid:** The Sigmoid function is typically used in the output layer of a binary classification model. It squashes any real-valued input into a range between 0 and 1.
    $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
    The output can be interpreted as a probability. A value close to 1 indicates a high probability of belonging to the positive class (in this case, the output being `1`), while a value close to 0 indicates a high probability of belonging to the negative class (output `0`).

#### 3\. Optimizer: Adam

The **Adam** (Adaptive Moment Estimation) optimizer is an efficient and widely used optimization algorithm. Unlike the standard Stochastic Gradient Descent (SGD), Adam adapts the learning rate for each weight in the network individually. It computes adaptive learning rates based on estimates of the first moment (the mean, like momentum) and the second moment (the uncentered variance) of the gradients. This often leads to faster convergence and better performance than standard SGD.

-----

### **Code**

```python
# Import necessary libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Sequential           # Sequential model for stacking layers linearly
from keras.layers import Dense, Activation    # Dense: fully connected layer; Activation: activation functions

# Define input (x) and output (y) data for logical AND operation
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 4 input combinations for 2 binary inputs
y = np.array([[0], [0], [0], [1]])             # Output is 1 only for [1, 1]

# Initialize a Sequential model
model = Sequential()

# Add a hidden layer with:
# - 16 neurons
# - ReLU activation function (to introduce non-linearity)
# - input_dim=2 for 2 input features
# - use_bias=False disables the bias term
model.add(Dense(16, input_dim=2, activation='relu', use_bias=False))

# Add output layer with:
# - 1 neuron (binary output)
# - Sigmoid activation function (squashes output between 0 and 1)
# - use_bias=False again disables the bias
model.add(Dense(1, activation='sigmoid', use_bias=False))

# Compile the model:
# - 'adam' optimizer for efficient training
# - 'mse' loss function (Mean Squared Error), though 'binary_crossentropy' is more common for classification
model.compile(optimizer='adam', loss='mse')

# Train the model using the input and output data
# for 350 epochs (iterations over the entire dataset)
model.fit(x, y, epochs=350)

# Print the model's prediction for each input combination
print("Prediction for [0, 0]:", model.predict(np.array([[0,0]])))
print("Prediction for [0, 1]:", model.predict(np.array([[0,1]])))
print("Prediction for [1, 0]:", model.predict(np.array([[1,0]])))
print("Prediction for [1, 1]:", model.predict(np.array([[1,1]])))
```

-----

### **Code Output**

**Training Log (abbreviated)**

```
Epoch 1/350
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 1s/step - loss: 0.2664
Epoch 2/350
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 152ms/step - loss: 0.2661
...
Epoch 349/350
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 166ms/step - loss: 0.2135
Epoch 350/350
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 182ms/step - loss: 0.2131
```

**Final Predictions**

```
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 150ms/step
Prediction for [0, 0]: [[0.5]]
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 115ms/step
Prediction for [0, 1]: [[0.3447347]]
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 109ms/step
Prediction for [1, 0]: [[0.33746344]]
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 114ms/step
Prediction for [1, 1]: [[0.6483854]]
```

-----

### **Code Explanation**

1.  **Import Libraries**: Import `tensorflow`, `numpy`, and `keras` components `Sequential`, `Dense`, and `Activation`.
2.  **Define Data**: The `x` and `y` NumPy arrays are created to represent the full truth table of the AND gate.
3.  **Create Model**: `model = Sequential()` initializes a linear stack of layers.
4.  **Add Hidden Layer**: The first `Dense` layer acts as a hidden layer. It has 16 neurons, uses `relu` activation, and is configured to accept inputs with 2 features (`input_dim=2`).
5.  **Add Output Layer**: The second `Dense` layer is the output layer. It has a single neuron to produce the final 0 or 1 prediction. The `sigmoid` activation function ensures the output is a value between 0 and 1.
6.  **Compile Model**: `model.compile()` configures the learning process. We use the `adam` optimizer and `mse` loss function.
7.  **Train Model**: `model.fit(x, y, epochs=350)` trains the network on the AND gate data for 350 iterations.
8.  **Predict**: `model.predict()` is called for each of the four possible inputs to see how the trained model behaves.

-----

### **Result**

The neural network was trained for 350 epochs. The final predictions for the four inputs were:

  * `[0, 0]` -\> `0.5`
  * `[0, 1]` -\> `0.34`
  * `[1, 0]` -\> `0.33`
  * `[1, 1]` -\> `0.65`

### **Inference**

The model has learned the general trend of the AND gate but has not perfected it. The prediction for `[1, 1]` (`0.65`) is the highest, and the predictions for the other inputs are lower, which aligns with the logic of an AND gate.

However, the results are not clean `0`s and `1`s. The model predicts `0.5` for `[0,0]` when it should be close to `0`. This imperfect learning is likely due to a combination of factors in the model's configuration:

1.  **Disabled Bias**: Setting `use_bias=False` makes it harder for the model to find the optimal decision boundary.
2.  **Loss Function**: While `mse` works, `binary_crossentropy` is the standard and more appropriate loss function for binary classification tasks, as it is designed to measure the distance between probability distributions.

The experiment successfully shows that a neural network can approximate a logical function, but it also highlights the importance of model configuration for achieving accurate results. With `binary_crossentropy` and enabled biases, the model would likely converge to more precise predictions.