<h1> Data Cleaning Essentials in Python for Google Colab</h1>

Data cleaning is a fundamental step within the data analysis process. In the context of Google Colab, you can leverage several Python libraries to effectively carry out data cleaning tasks. Below are key commands, libraries, and code snippets that can be employed to execute data cleaning procedures in Python within the Google Colab environment:

<h2>Essential Libraries:</h2>

```
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical operations
```
<h2>Data Loading:</h2>


```
df = pd.read_csv('file.csv')  # Load data from a CSV file
```
<h2>Initial Exploration:</h2>

```
df.head()        # Display the first few rows of the DataFrame
df.info()        # General information about the DataFrame
df.describe()    # Statistical summary of the DataFrame
df.shape         # Dimensions of the DataFrame (rows, columns)
```
<h2>Handling Missing Values:</h2>

```
df.isnull().sum()         # Count null values per column
df.dropna()               # Drop rows with null values
df.fillna(value)          # Fill null values with a specific value
df.interpolate(method)    # Interpolate null values based on nearby values
```

<h2>Duplicate Data Handling:</h2>

```
df.duplicated().sum()     # Count duplicates in the DataFrame
df.drop_duplicates()      # Drop duplicate rows
```

<h2>Column Management:</h2>

```
df.rename(columns={'old_name': 'new_name'}, inplace=True)  # Rename columns
df['column_name'].astype(dtype)  # Change data type of a column
```

<h2>Column Removal:</h2>

```
df.drop(columns=['column_name'], inplace=True)  # Drop columns
```

<h2>Data Transformation:</h2>

```
df['new_column'] = df['column_name'].apply(func)  # Create a new column using a function
df['new_column'] = np.log(df['column_name'])      # Apply mathematical operations
```


<h1>Scaling and Normalization</h1>:

Scaling and normalization are techniques used to adjust numerical features to a specific range, which can help improve the performance of certain machine learning algorithms.

```
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Create a sample DataFrame (As an example)
data = {'Feature1': [10, 20, 30, 40, 50],
        'Feature2': [0.1, 0.5, 1.0, 1.5, 2.0]}
df = pd.DataFrame(data)

# Standard Scaling (Z-score)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
print("DataFrame after scaling (Z-score):")
print(df_scaled)

# Min-Max Normalization in the range [0, 1]
minmax_scaler = MinMaxScaler()
df_normalized = minmax_scaler.fit_transform(df)
print("\nDataFrame after min-max normalization in the range [0, 1]:")
print(df_normalized)
```

<h2>Explanations:</h2>

<li> StandardScaler: Applies standard scaling (Z-score) to the features. This means each feature is transformed to have a mean of 0 and a standard deviation of 1. It's useful when features have non-Gaussian distributions.

<li> MinMaxScaler: Performs normalization of features to be within the range [0, 1]. This is helpful when features have different ranges and you want them to have a common range.

<h2>Encoding Categorical Variables into Dummy Variables:</h2>

Encoding categorical variables into dummy variables is a technique to convert categorical variables into binary numeric variables, which are more suitable for many machine learning algorithms.

```
# Create a sample DataFrame with a categorical variable (As an example)
data = {'Category': ['A', 'B', 'A', 'C', 'B']}
df = pd.DataFrame(data)

# Encoding categorical variables into dummy variables
df_encoded = pd.get_dummies(df, columns=['Category'])  # One-hot encoding
print("\nDataFrame after encoding into dummy variables:")
print(df_encoded)
```

<h2>Explanation:</h2>

pd.get_dummies: This pandas method takes a categorical column and creates new binary columns for each unique category. Each new column represents the presence or absence of that category in the corresponding record.
In summary, scaling and normalization adjust numerical features to specific ranges for better algorithm performance, and encoding categorical variables into dummy variables transforms categorical variables into a suitable format for machine learning analysis.

Another way:

```
df['encoded_column'] = pd.Categorical(df['categorical_column']).codes  # Categorical encoding
```
