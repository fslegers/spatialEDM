import os
import pandas as pd
import numpy as np

def calculate_rmse(observed, predicted):
    return np.sqrt(np.mean((observed - predicted)**2))

def calculate_confidence_intervals(observed, predicted):
    n = len(observed)
    std_dev = np.std(np.abs(observed - predicted))
    mean = np.mean(np.abs(observed - predicted))
    return mean - 2*std_dev, mean + 2*std_dev

# Define path to folder
folder_path  = 'C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten'
folder_path = folder_path + '/05-03-24'

# Load all CSV files from the folder into a pandas DataFrame
dfs = []
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
df = pd.concat(dfs)

# Group by unique combinations of hor, obs_noise, and init_var
grouped = df.groupby(['hor', 'obs_noise', 'init_var'])

# Create lists to store results
hor_list = []
obs_noise_list = []
init_var_list = []
rmse_list = []
lower_ci_list = []
upper_ci_list = []

# Iterate over groups and calculate RMSE
for group_name, group_df in grouped:
    hor, obs_noise, init_var = group_name
    rmse = calculate_rmse(group_df['obs'], group_df['pred_smap'])
    lower, upper = calculate_confidence_intervals(group_df['obs'], group_df['pred_smap'])

    # Append results to respective lists
    hor_list.append(hor)
    obs_noise_list.append(obs_noise)
    init_var_list.append(init_var)
    rmse_list.append(rmse)
    lower_ci_list.append(lower)
    upper_ci_list.append(upper)

# Create DataFrame from lists
results = pd.DataFrame({
    'hor': hor_list,
    'obs_noise': obs_noise_list,
    'init_var': init_var_list,
    'RMSE': rmse_list,
    'lower': lower_ci_list,
    'upper': upper_ci_list
})

# Save results to a new CSV file
path_name = folder_path + "/summarized/results.csv"
results.to_csv(path_name, index=False)