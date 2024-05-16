import pandas as pd

# Read the first CSV file containing patches names
patches_df = pd.read_csv('zenodo_patches_names.csv')

# Read the second CSV file containing selected data
selected_df = pd.read_csv('selected_data.csv')

# Extract the 'case_id' column from patches_df based on the pattern
patches_df['case_id'] = patches_df['PATIENT'].str.split('-').str[2]

# Extract the 'slide_id' column from patches_df based on the pattern
patches_df['slide_id'] = patches_df['PATIENT'].str.split('-').str[2]

# Extract the 'case_id' and 'MSIStatus' columns from selected_df
selected_df['case_id'] = selected_df['PATIENT'].str.split('-').str[2]

# Filter patches_df to include only records where the case_id exists in selected_df
patches_df = patches_df[patches_df['case_id'].isin(selected_df['case_id'])]

# Create a new column 'label' in patches_df based on matching 'case_id' and 'MSIStatus'
patches_df['label'] = patches_df['case_id'].isin(selected_df.loc[selected_df['isMSIH'] == 'MSIH', 'case_id']).astype(int)

# Extract the 'case_id' column from patches_df based on the pattern
patches_df['name'] = patches_df['PATIENT']

# Extract the 'slide_id' column from patches_df based on the pattern
patches_df['rpath'] = patches_df['PATIENT']

patches_df.drop(columns=['PATIENT'], inplace=True)

patches_df.drop(columns=['case_id', 'slide_id'], inplace=True)
patches_df = patches_df.iloc[:, [2,1,0]]
patches_df = patches_df.iloc[:, [1,0,2]]
# Save the modified DataFrame patches_df to a new CSV file
patches_df.to_csv('dataset.csv', index=False)

print("New CSV file created with label column.")
