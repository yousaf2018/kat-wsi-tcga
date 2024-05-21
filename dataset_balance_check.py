import pandas as pd

# Read the CSV file
df = pd.read_csv("dataset.csv")

# Count occurrences of each label
label_counts = df['label'].value_counts()

# Print the counts
print("Non MSI count:", label_counts.get(0, 0))
print("MSI count:", label_counts.get(1, 0))
