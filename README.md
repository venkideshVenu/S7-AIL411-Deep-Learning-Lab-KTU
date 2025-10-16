# AIL411: Deep Learning Lab

This repository contains the experiments and solutions for the **AIL411: Deep Learning Lab** course, part of the B.Tech CSE (Artificial Intelligence) curriculum. The lab focuses on providing hands-on experience with fundamental and advanced deep learning algorithms and their applications.

## Course Preamble

This course aims to offer students hands-on experience on deep learning algorithms. Students will be able to familiarize basic python packages for deep learning, computer vision concepts for deep learning, sequence modelling and recurrent neural network. This course helps the learners to enhance the capability to design and implement a deep learning architecture for a real time application.

## Course Outcomes

Upon completion of this course, students will be able to:

- **CO 1:** Implement advanced machine learning concepts using python.
- **CO 2:** Apply basic data pre-processing and tuning techniques.
- **CO 3:** Experiment behaviour of neural networks and CNN on datasets.
- **CO 4:** Design and Implement sequence modelling schemes.
- **CO 5:** Implement auto encoders on standard datasets and analyse the performance.

## Repository Structure

Each experiment is organized into its own directory, containing:

- A **Jupyter Notebook** (`.ipynb`) with the Python code implementation.
- A detailed **Markdown file** (`.md`) that includes the aim, algorithm, theory, code explanation, and results for the experiment.
- Any necessary **data files** or **output images**.

```
📦 AIL411-Deep-Learning-Lab-KTU
├── 📁 00 Additional Programs/                       # Extra experiments and advanced implementations
│   └── 01 BiLSTM Chatbot.ipynb/.md
├── 📁 00 Inputs/                                    # Input data files
├── 📁 00 Outputs/                                   # Generated output images
├── 📁 01 Familiarization of Python Packages/       # Basic Python & ML experiments
│   ├── 1_Linear_Regression.ipynb/.md
│   ├── 2 Image Enhancement.ipynb/.md
│   ├── 3 Implement Y = X.ipynb/.md
│   └── 4 Implement AND GATE.ipynb/.md
├── 📋 02 Outlier Management.ipynb/.md               # Data preprocessing
├── 📋 03 FeedForwardNetwork.ipynb/.md               # Neural Networks
├── 📋 04 Optimization and Weight Initialization Techniques.ipynb/.md
├── 📋 05 Digit Classification using CNN.ipynb/.md   # Computer Vision
├── 📋 06 Digit Classification using VGGnet-19.ipynb/.md
├── 📋 07 Review Classification IMDB.ipynb/.md       # NLP & RNN
├── 📋 08 LSTM vs GRU Performance Analysis.ipynb/.md # Advanced RNN
├── 📋 09 NIFTY-50 Time Series Forecasting.ipynb/.md # Time Series Analysis
├── 📋 10 English–Hindi Translation Using Shallow Autoencoder–Decoder.ipynb/.md # Machine Translation
├── 📄 Syllabus.pdf                                 # Course syllabus
└── 📄 README.md                                    # This file
```

## List of Experiments

Here is a list of the mandatory experiments completed as part of this lab:

| Exp. No. | Experiment Title                                                                                         | Status           | Notebook                                                                                        | Documentation                                                                            |
| -------- | -------------------------------------------------------------------------------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **1.1**  | Program to implement Linear Regression                                                                   | ✅ **Completed** | [📓 Notebook](./01%20Familiarization%20of%20Python%20Packages/1_Linear_Regression.ipynb)        | [📄 Docs](./01%20Familiarization%20of%20Python%20Packages/1_Linear_Regression.md)        |
| **1.2**  | Image Enhancement Techniques                                                                             | ✅ **Completed** | [📓 Notebook](./01%20Familiarization%20of%20Python%20Packages/2%20Image%20Enhancement.ipynb)    | [📄 Docs](./01%20Familiarization%20of%20Python%20Packages/2%20Image%20Enhancement.md)    |
| **1.3**  | Implement Y = X Neural Network                                                                           | ✅ **Completed** | [📓 Notebook](./01%20Familiarization%20of%20Python%20Packages/3%20Implement%20Y%20=%20X.ipynb)  | [📄 Docs](./01%20Familiarization%20of%20Python%20Packages/3%20Implement%20Y%20=%20X.md)  |
| **1.4**  | Implement AND GATE with Neural Network                                                                   | ✅ **Completed** | [📓 Notebook](./01%20Familiarization%20of%20Python%20Packages/4%20Implement%20AND%20GATE.ipynb) | [📄 Docs](./01%20Familiarization%20of%20Python%20Packages/4%20Implement%20AND%20GATE.md) |
| **2**    | Data pre-processing operations such as outliers and/or inconsistent data value management                | ✅ **Completed** | [📓 Notebook](./02%20Outlier%20Management.ipynb)                                                | [📄 Docs](./02%20Outlier%20Management.md)                                                |
| **3**    | Implement Feed forward neural network with three hidden layers for classification on CIFAR-10 dataset    | ✅ **Completed** | [📓 Notebook](./03%20FeedForwardNetwork.ipynb)                                                  | [📄 Docs](./03%20FeedForwardNetwork.md)                                                  |
| **4**    | Analyse the impact of optimization and weight initialization techniques (Xavier, Kaiming, Dropout, etc.) | ✅ **Completed** | [📓 Notebook](./04%20Optimization%20and%20Weight%20Initialization%20Techniques.ipynb)           | [📄 Docs](./04%20Optimization%20and%20Weight%20Initialization%20Techniques.md)           |
| **5**    | Digit classification using CNN architecture for MNIST dataset                                            | ✅ **Completed** | [📓 Notebook](./05%20Digit%20Classification%20using%20CNN.ipynb)                                | [📄 Docs](./05%20Digit%20Classification%20using%20CNN.md)                                |
| **6**    | Digit classification using pre-trained networks like VGGnet-19 for MNIST dataset                         | ✅ **Completed** | [📓 Notebook](./06%20Digit%20Classification%20using%20VGGnet-19.ipynb)                          | [📄 Docs](./06%20Digit%20Classification%20using%20VGGnet-19.md)                          |
| **7**    | Implement a simple RNN for review classification using IMDB dataset                                      | ✅ **Completed** | [📓 Notebook](./07%20Review%20Classification%20IMDB.ipynb)                                      | [📄 Docs](./07%20Review%20Classification%20IMDB.md)                                      |
| **8**    | Analyse and visualize the performance change while using LSTM and GRU instead of simple RNN              | ✅ **Completed** | [📓 Notebook](./08%20LSTM%20vs%20GRU%20Performance%20Analysis.ipynb)                            | [📄 Docs](./08%20LSTM%20vs%20GRU%20Performance%20Analysis.md)                            |
| **9**    | Implement time series forecasting prediction for NIFTY-50 dataset                                        | ✅ **Completed** | [📓 Notebook](./09%20NIFTY-50%20Time%20Series%20Forecasting.ipynb)                              | [📄 Docs](./09%20NIFTY-50%20Time%20Series%20Forecasting.md)                              |
| **10**   | Implement a shallow auto encoder and decoder network for machine translation                             | ✅ **Completed** | [📓 Notebook](./10%20English–Hindi%20Translation%20Using%20Shallow%20Autoencoder–Decoder.ipynb) | [📄 Docs](./10%20English–Hindi%20Translation%20Using%20Shallow%20Autoencoder–Decoder.md) |

## Additional Experiments

Beyond the mandatory curriculum, this repository also includes advanced implementations and explorations:

| Exp. No. | Experiment Title                                   | Status           | Notebook                                                                | Documentation                                                    |
| -------- | -------------------------------------------------- | ---------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **A.1**  | BiLSTM-based Conversational Chatbot Implementation | ✅ **Completed** | [📓 Notebook](./00%20Additional%20Programs/01%20BiLSTM%20Chatbot.ipynb) | [📄 Docs](./00%20Additional%20Programs/01%20BiLSTM%20Chatbot.md) |

> **Note:** This README will be updated as more additional experiments are added.

## Experiment Details

### 🔰 Familiarization of Python Packages

- **Linear Regression:** Implementation from scratch using scikit-learn for height-weight prediction
- **Image Enhancement:** Computer vision techniques including histogram equalization, morphological operations
- **Basic Neural Networks:** Simple implementations including Y=X mapping and AND gate logic

### 📊 Data Preprocessing

- **Outlier Management:** Techniques for detecting and handling outliers in datasets
- **Data cleaning and normalization methods**

### 🧠 Neural Networks & Deep Learning

- **Feed-Forward Networks:** Multi-layer perceptrons for CIFAR-10 classification
- **Optimization Techniques:** Comparison of Xavier, Kaiming initialization, Dropout, and Batch Normalization

### 👁️ Computer Vision

- **CNN Implementation:** Custom CNN architecture for MNIST digit classification
- **Transfer Learning:** Using pre-trained VGGNet-19 for enhanced digit classification

### 📝 Natural Language Processing

- **RNN for Sentiment Analysis:** Simple RNN implementation for IMDB movie review classification
- **Advanced RNN architectures:** LSTM and GRU comparison for performance analysis
- **Machine Translation:** English-Hindi translation using shallow autoencoder-decoder networks

### 📈 Time Series & Advanced Applications

- **Time Series Forecasting:** NIFTY-50 stock prediction using deep learning models
- **Sequence Modeling:** Advanced RNN architectures for temporal data analysis

### 🤖 Additional Advanced Implementations

- **Conversational AI:** BiLSTM-based chatbot for natural language conversation
- **Advanced NLP:** Cutting-edge deep learning applications beyond the core curriculum

## Getting Started

To run these experiments on your local machine, follow these steps:

### Prerequisites

- **Python 3.8+** and **pip** installed
- **Jupyter Notebook** or **JupyterLab**
- **Git** (for cloning the repository)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/S7-AIL411-Deep-Learning-Lab-KTU.git
   cd S7-AIL411-Deep-Learning-Lab-KTU
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Required Dependencies

The experiments use the following core libraries:

```python
# Deep Learning & Machine Learning
tensorflow>=2.10.0
keras>=2.10.0
scikit-learn>=1.1.0
numpy>=1.21.0
pandas>=1.4.0

# Computer Vision
opencv-python>=4.6.0
Pillow>=9.0.0

# Data Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Jupyter Environment
jupyter>=1.0.0
ipykernel>=6.0.0
```

### Running the Experiments

1. **Navigate to the project directory:**

   ```bash
   cd S7-AIL411-Deep-Learning-Lab-KTU
   ```

2. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

3. **Open any experiment notebook** (`.ipynb` file) to view and run the code for that experiment.

4. **Run cells sequentially** using `Shift + Enter` or use "Run All" from the Cell menu.

## Technologies Used

### Programming Language

- **Python 3.8+**

### Core Libraries

#### Deep Learning & ML

- **TensorFlow & Keras:** For building and training deep learning models
- **Scikit-learn:** For traditional ML algorithms and utility functions
- **NumPy:** For numerical operations and array manipulation

#### Data Processing & Visualization

- **Pandas:** For data manipulation and analysis
- **Matplotlib & Seaborn:** For data visualization and plotting
- **OpenCV:** For computer vision and image processing tasks

#### Development Environment

- **Jupyter Notebook:** For interactive development and documentation

### Datasets Used

- **CIFAR-10:** 32x32 color images in 10 classes (60,000 images)
- **MNIST:** 28x28 grayscale handwritten digits (70,000 images)
- **IMDB Movie Reviews:** Text data for sentiment analysis (50,000 reviews)
- **Custom datasets:** Height-weight data, clock images, etc.

## Key Learning Outcomes

Through these experiments, you will gain practical experience in:

### 🎯 Machine Learning Fundamentals

- Linear regression implementation and evaluation
- Data preprocessing and feature engineering
- Model evaluation metrics and validation techniques

### 🧠 Deep Learning Architectures

- Feed-forward neural networks (Multi-layer Perceptrons)
- Convolutional Neural Networks (CNNs) for image classification
- Recurrent Neural Networks (RNNs) for sequential data processing

### ⚙️ Optimization & Regularization

- Weight initialization techniques (Xavier/Glorot, He/Kaiming)
- Regularization methods (Dropout, Batch Normalization)
- Optimizer comparison (SGD, Adam, RMSprop)

### 👁️ Computer Vision

- Image preprocessing and augmentation
- CNN architecture design and implementation
- Transfer learning with pre-trained models

### 📝 Natural Language Processing

- Text preprocessing and tokenization
- Sequence modeling with RNNs
- Sentiment analysis and text classification

### 📊 Data Science Skills

- Outlier detection and management
- Data visualization and interpretation
- Performance analysis and model comparison

## Performance Highlights

### Model Accuracies Achieved

- **Linear Regression:** Low MSE on height-weight prediction
- **CIFAR-10 FFNN:** ~45-50% accuracy (3 hidden layers)
- **MNIST CNN:** ~98-99% accuracy (custom architecture)
- **MNIST VGGNet-19:** ~99%+ accuracy (transfer learning)
- **IMDB RNN:** ~73-74% accuracy (sentiment classification)
- **LSTM vs GRU:** Comparative analysis showing improved performance over simple RNN
- **NIFTY-50 Forecasting:** Time series prediction with deep learning models
- **English-Hindi Translation:** Autoencoder-decoder based machine translation
- **BiLSTM Chatbot:** Advanced conversational AI implementation

### Optimization Results

- **Xavier vs. Kaiming:** Demonstrated initialization impact on convergence
- **L2 Regularization:** Achieved 50.38% accuracy (best performance in optimization study)
- **Dropout effectiveness:** Reduced overfitting in deep networks
- **Batch Normalization:** Accelerated training and improved stability
- **Advanced RNN Architectures:** LSTM and GRU showing superior performance over simple RNN

## Reference Books

1. **Deep Learning with Python** by François Chollet, Manning, 2021
2. **Deep Learning** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville, MIT Press, 2016
3. **Neural Networks and Deep Learning** by Charu C. Aggarwal, Springer, 2018
4. **Hands-On Machine Learning** by Aurélien Géron, O'Reilly Media, 2019

## Contributing

This repository is part of academic coursework. However, suggestions for improvements are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is part of the AIL411 Deep Learning Lab course curriculum. Please respect academic integrity guidelines when using this code.

## Contact

For questions or clarifications regarding the experiments:

- **Course:** AIL411 - Deep Learning Lab
- **Program:** B.Tech CSE (Artificial Intelligence)
- **University:** Kerala Technological University (KTU)

---

**Happy Learning! 🚀**

_Last Updated: October 2025_
