# import numpy as np
# import pandas as pd

# # Load the .npy file
# data_y = np.load("y_data_final.npy")  # Replace with your actual file path
# data_X = np.load("X_data_final.npy")

# # Print the shape to understand its structure
# print(f"Outputs Data shape: {data_y.shape}")  # Expected: (num_samples, 21, num_features)
# print(f"Input Data shape: {data_X.shape}")  # Expected: (num_samples, 21, num_features)



# # Get number of samples
# num_samples = data.shape[0]
# print(f"Number of samples: {num_samples}")


# top5 = data[:5]  # Shape: (5, time_steps, features)

# # Save to text file in a readable format
# with open("top5_y_data.txt", "w") as f:
#     for i, sample in enumerate(top5):
#         f.write(f"Sample {i}:\n")
#         np.savetxt(f, sample, fmt="%.4f")
#         f.write("\n" + "-"*40 + "\n")



# column_means = data[:, 3:].mean(axis=0)

# # Print the result
# print("Column means from column 3 onwards:\n", column_means)







# df = pd.read_parquet('climax_processed_NA.parquet')
# print(df.head())
# print(df.shape)  # This will print (number of rows, number of columns)

# def summarize_parquet_columns(parquet_file):
#     # Load the DataFrame
#     df = pd.read_parquet(parquet_file)

#     # Drop non-numeric columns (optional, depending on your dataset)
#     df_numeric = df.select_dtypes(include='number')

#     # Calculate mean and variance
#     summary = pd.DataFrame({
#         'Mean': df_numeric.mean(),
#         'Variance': df_numeric.var()
#     })

#     # Reset index to get column names as a column
#     summary.reset_index(inplace=True)
#     summary.rename(columns={'index': 'Feature'}, inplace=True)

#     print(summary.to_string(index=False))

#     return summary


# # Example usage
# summary_df = summarize_parquet_columns('climax_processed_NA.parquet')




# ________________________________________________________________________

# import numpy as np

# # Replace with the path to your .npz file
# file_path = '/home/vincent.guo/ClimaX2.0/data/5.625deg_npzAllVars/train/2015_4.npz'

# # Load the .npz file
# data = np.load(file_path)

# # Print the keys and the size of each value
# print("Keys and sizes in the .npz file:")
# for key in data.files:
#     value = data[key]
#     print(f"{key}: shape = {value.shape}, dtype = {value.dtype}")



import matplotlib.pyplot as plt

# Example data
x = [2, 3, 5, 7, 14, 21]
y1 = [0.5812, 0.5824, 0.6129, 0.6152, 0.6572, 0.6598]                      # Line 1: y = x
y2 = [0.5805, 0.5848, 0.6098, 0.6155, 0.6601, 0.6640]                   # Line 2: y = x^2
y3 = [0.5963, 0.6058, 0.6280, 0.6403, 0.6834, 0.6927]                 # Line 3: y = √x

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='Hyperparameter set 1', marker='o', color='blue', markerfacecolor='blue', markeredgecolor='blue')
plt.plot(x, y2, label='Hyperparameter set 2', marker='s', color='red', markerfacecolor='red', markeredgecolor='red')
plt.plot(x, y3, label='Hyperparameter set 3', marker='^', color='green', markerfacecolor='green', markeredgecolor='green')

# Add labels, legend, and title
plt.xlabel('t value')
plt.xticks(x)
plt.ylabel('Model Performace (w_rmse)')
# plt.title('Three Series on the Same Plot')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig('tvperf.png', dpi=300, bbox_inches='tight')

# Optionally show the plot
plt.show()
