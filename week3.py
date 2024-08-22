import pandas as pd

# File paths for the Boning and Slicing datasets
file_path_boning = r'C:\week 3\ampc2\Boning.csv'
file_path_slicing = r'C:\week 3\ampc2\Slicing.csv'

# Load the datasets
data_boning = pd.read_csv(file_path_boning)
data_slicing = pd.read_csv(file_path_slicing)

# Define the correct columns to select
selected_columns = [
    'Frame',  # Frame column
    'Right Lower Leg x', 'Right Lower Leg y', 'Right Lower Leg z',  # 3 columns for Right Hip/Leg
    'Left Lower Leg x', 'Left Lower Leg y', 'Left Lower Leg z'  # 3 columns for Left Hip/Leg
]

# Extract the relevant columns from both datasets and add the class label
boning_data = data_boning[selected_columns].copy()
boning_data['class'] = 0  # Label for boning

slicing_data = data_slicing[selected_columns].copy()
slicing_data['class'] = 1  # Label for slicing

# Combine both datasets into one DataFrame
combined_data = pd.concat([boning_data, slicing_data], ignore_index=True)

# Preview the combined data to ensure correctness
print(combined_data.head())

# Save the combined data to a new CSV file
output_file_path = r'C:\week 3\ampc2\combined_data.csv'
combined_data.to_csv(output_file_path, index=False)
