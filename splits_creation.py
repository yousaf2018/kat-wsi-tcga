import pandas as pd

# Read the dataset
df = pd.read_csv('dataset.csv')

# Split the dataset into train and test with 70-30 ratio
train_ratio = 0.7
test_ratio = 0.3

# Get the number of rows for train and test data
train_size = int(len(df) * train_ratio)
test_size = len(df) - train_size

# Split the dataset
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Save the split data to CSV files
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)
