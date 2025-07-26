# **Experiment 02: Data Pre-processing and Outlier Management**

### **Aim**

To perform data pre-processing by identifying and managing outliers in a dataset. The experiment demonstrates how to calculate basic descriptive statistics, detect outliers using the Z-score and Interquartile Range (IQR) methods, and handle them by removal or capping.

### **Algorithm**

1.  **Start**
2.  Import necessary libraries (`statistics`, `numpy`, `pandas`, `matplotlib`, `seaborn`).
3.  Define a sample dataset with a potential outlier.
4.  Calculate and print basic statistical measures: mean, median, mode, variance, and standard deviation for the original data.
5.  **Z-Score Method:**
      * Define a function `detect_outliers_zscore` to identify outliers.
      * Calculate the Z-score for each data point.
      * Identify any data point with a Z-score greater than a threshold (e.g., 3) as an outlier.
      * Create a new dataset by removing the identified outliers.
      * Compare the statistical measures of the original and the new dataset.
6.  **IQR Method:**
      * Define a function `detect_outliers_iqr` to identify outliers.
      * Calculate the first (Q1) and third (Q3) quartiles.
      * Compute the Interquartile Range (IQR).
      * Define the lower and upper bounds (Q1 - 1.5*IQR and Q3 + 1.5*IQR).
      * Identify any data point outside these bounds as an outlier and print it.
7.  **Capping/Clipping Method:**
      * Calculate the 10th and 90th percentiles of the data.
      * Create a new array where values below the 10th percentile are replaced by the 10th percentile value, and values above the 90th percentile are replaced by the 90th percentile value.
      * Print the new, capped array.
8.  **Stop**

### **Inputs and Outputs**

  * **Input:** A list of numerical data: `[15, 101, 18, 7, 13, 16, 11, 21, 5, 15, 10, 9]`
  * **Outputs:**
    1.  Basic statistics of the original data.
    2.  A comparison of statistics for the data with and without the outlier removed via the Z-score method.
    3.  A list of outliers identified using the IQR method.
    4.  A new array showing the result of capping the data at the 10th and 90th percentiles.

### **Theory**

#### 1\. Descriptive Statistics

Descriptive statistics are summary statistics that quantitatively describe or summarize features of a collection of information.

  * **Mean**: The average of all data points. It is sensitive to outliers.
  * **Median**: The middle value in a sorted dataset. It is robust to outliers.
  * **Mode**: The most frequently occurring value in a dataset.
  * **Variance**: A measure of how spread out the data is. It is the average of the squared differences from the Mean.
  * **Standard Deviation**: The square root of the variance, representing the average amount of variability in your dataset.

#### 2\. Outliers

An **outlier** is a data point that differs significantly from other observations. Outliers can be caused by measurement errors or may indicate novel findings in the data. They can severely distort statistical analyses and machine learning models.

#### 3\. Z-Score Method

The Z-score (or standard score) measures how many standard deviations a data point is from the mean of the dataset. A common rule of thumb is to consider a data point an outlier if its Z-score is greater than 3 or less than -3.
The formula is:
$$Z = \frac{(x - \mu)}{\sigma}$$
Where:

  * $x$ is the data point.
  * $\\mu$ is the mean of the dataset.
  * $\\sigma$ is the standard deviation of the dataset.

#### 4\. Interquartile Range (IQR) Method

The IQR method is a robust statistical technique for identifying outliers. The IQR is the range between the first quartile (Q1, 25th percentile) and the third quartile (Q3, 75th percentile).
A data point is considered an outlier if it falls outside the following range:

  * **Lower Bound**: $Q1 - 1.5 \\times IQR$
  * **Upper Bound**: $Q3 + 1.5 \\times IQR$
    This method is generally preferred over the Z-score method because it is not influenced by the extreme values (outliers) themselves.

#### 5\. Capping and Clipping

Capping is a technique for handling outliers where we replace the outlier values with a certain percentile value. For instance, any value above the 90th percentile is replaced by the 90th percentile value, and any value below the 10th percentile is replaced by the 10th percentile value. This is also known as **winsorization**. It helps to reduce the effect of outliers without completely removing the data points.

-----

### **Code**

```python
import statistics as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data with a clear outlier (101)
data = [15, 101, 18, 7, 13, 16, 11, 21, 5, 15, 10, 9]

# --- 1. Basic Statistics ---
print('Mean           : ' ,st.mean(data))
print('Median         : ', st.median(data))
print('Mode           : ', st.mode(data))
print('Varience       : ', st.variance(data))
print('Std Deviation  : ', st.stdev(data))

# --- 2. Removing outliers using Z-score ---
outliers_z = []
def detect_outliers_zscore(data):
    thres = 3
    mean = np.mean(data)
    std = np.std(data)
    for i in data:
        z_score = (i-mean) / std
        if np.abs(z_score) > thres:
            outliers_z.append(i)
    return outliers_z

# Detect and create a new list without the outliers
out = detect_outliers_zscore(data)
NewData = [i for i in data if i not in out]

# Compare stats with and without the outlier
print("\n                   with Outlier          without Outlier")
print("Mean           : ", round(st.mean(data),2), "\t\t", round(st.mean(NewData),2))
print('Median         : ', round(st.median(data),2), "\t\t\t", round(st.median(NewData),2))
print('Mode           : ', round(st.mode(data),2), "\t\t\t", round(st.mode(NewData),2))
print('Variance       : ', round(st.variance(data),2), "\t\t", round(st.variance(NewData),2))
print('Std deviation  : ' , round(st.stdev(data),2), "\t\t\t", round(st.stdev(NewData),2))

# --- 3. Using IQR ---
outliers_iqr = []
def detect_outliers_iqr(data):
    data = sorted(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    IQR = q3 - q1
    lwr_bound = q1 - (1.5 * IQR)
    upr_bound = q3 + (1.5 * IQR)
    for i in data:
        if i < lwr_bound or i > upr_bound:
            outliers_iqr.append(i)
    return outliers_iqr

sample_outliers = detect_outliers_iqr(data)
print("\nOutliers from IQR method: ", sample_outliers)

# --- 4. Capping the data ---
tenth_percentile = np.percentile(data, 10)
ninetieth_percentile = np.percentile(data, 90)

# Replace values outside the 10-90 percentile range
b = np.where(data < tenth_percentile, tenth_percentile, data)
b = np.where(b > ninetieth_percentile, ninetieth_percentile, b)
print("\n10 %", tenth_percentile, " \n90%", ninetieth_percentile, "\nNew array:", b)
```

-----

### **Code Output**

```
Mean           :  20.083333333333332
Median         :  14.0
Mode           :  15
Varience       :  670.6287878787879
Std Deviation  :  25.896501460212495

                   with Outlier          without Outlier
Mean           :  20.08 		 12.73
Median         :  14.0 			 13
Mode           :  15 			 15
Variance       :  670.63 		 23.42
Std deviation  :  25.9 			 4.84

Outliers from IQR method:  [101]

10 % 7.2  
90% 20.700000000000003 
New array: [15.  20.7 18.   7.2 13.  16.  11.  20.7  7.2 15.  10.   9. ]
```

-----

### **Code Explanation**

1.  **Initial Statistics**: The code first calculates and prints the mean, median, mode, variance, and standard deviation of the raw data. This serves as a baseline to see the effect of the outlier `101`.
2.  **Z-Score Detection**: The `detect_outliers_zscore` function iterates through the data. For each point, it calculates the Z-score. If the absolute Z-score is greater than 3 (a standard threshold), the point is flagged as an outlier.
3.  **Outlier Removal & Comparison**: A new list, `NewData`, is created by excluding the outlier found via the Z-score method. The code then prints a side-by-side comparison of the key statistics for the original and cleaned data, showing the significant impact of removing the outlier.
4.  **IQR Detection**: The `detect_outliers_iqr` function first sorts the data, then uses `np.percentile` to find Q1 and Q3. It calculates the IQR and determines the valid data range. Any number outside this range is identified as an outlier.
5.  **Capping**: The code finds the 10th and 90th percentile values. The `np.where` function is then used twice: first to replace any value smaller than the 10th percentile with the 10th percentile value, and second to replace any value larger than the 90th percentile with the 90th percentile value.

-----

### **Result**

  * The initial statistics were heavily skewed by the outlier (`101`). The mean (`20.08`) was much higher than the median (`14.0`), and the variance and standard deviation were extremely large.
  * After removing the outlier using the Z-score method, the statistics became much more representative of the "typical" values. The mean dropped to `12.73`, much closer to the new median of `13`. The variance plummeted from `670.63` to `23.42`.
  * The IQR method also successfully identified `101` as the sole outlier.
  * The capping method replaced the outlier `101` with the 90th percentile value (`20.7`) and the smallest value `5` with the 10th percentile value (`7.2`), creating a new dataset with reduced extreme values.

### **Inference**

This experiment clearly demonstrates the profound impact that outliers can have on descriptive statistics. The **mean** and **variance** are particularly sensitive to extreme values. Removing the outlier provided a more accurate and stable representation of the central tendency and spread of the dataset.

Both the **Z-score** and **IQR** methods were effective in identifying the outlier in this simple dataset. The choice between them often depends on the distribution of the data and the preference for a method that is robust to the influence of the outliers themselves (favoring IQR). Capping provides a less drastic alternative to removal, preserving the data point but mitigating its extreme influence.