import matplotlib.pyplot as plt
import numpy as np


# Example data
x = [1, 3, 5, 7]
Z500 = [433.03, 1000.68, 1072.36, 1084.76]
T850 = [2.014, 3.755, 3.857, 3.882]                  
T2M = [1.350, 2.640, 2.697, 2.739]                   
U10 = [2.976, 4.874, 5.036, 5.054]
V10 = [3.357, 5.407, 5.486, 5.516]
AVG = [88.545, 203.471, 217.887, 220.390]



Z500_1 = [301.87, 406.87, 510.12, 603.69]
T850_1 = [2.063, 2.785, 3.657, 4.048]   
T2M_1 = [2.354, 3.083, 3.872, 4.008]            
U10_1 = [1.961, 2.579, 3.024, 3.375]         
V10_1 = [2.148, 2.899, 3.472, 3.916]
AVG_1 = [62.079, 83.643, 104.829, 123.807]


# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, V10, label='ClimaX', marker='o', color='blue', markerfacecolor='blue', markeredgecolor='blue')
plt.plot(x, V10_1, label='FC V1', marker='s', color='red', markerfacecolor='red', markeredgecolor='red')

# plt.plot(x, T850, label='T850', marker='o', color='blue', markerfacecolor='blue', markeredgecolor='blue')
# plt.plot(x, T2M, label='T2M', marker='s', color='red', markerfacecolor='red', markeredgecolor='red')
# plt.plot(x, U10, label='U10', marker='^', color='green', markerfacecolor='green', markeredgecolor='green')
# plt.plot(x, V10, label='V10', marker='d', color='black', markerfacecolor='black', markeredgecolor='black')


# Add labels, legend, and title
plt.xlabel('Prediction Window (Days)', fontsize = 16)
plt.xticks(x, fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('V10 WRMSE', fontsize = 16)
# plt.title('Three Series on the Same Plot')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig('V10.png', dpi=300, bbox_inches='tight')

# Optionally show the plot
plt.show()


# ________________________________________________

# # Example data
# # x = [10, 20, 30, 50, 70, 71, 73, 75, 85, 100]
# # y1 = [96.08468, 91.83526, 89.63392, 89.29365, 88.546097, 94.23267, 94.661125, 96.470657, 95.491066, 94.857666]
# x = [20, 30, 50, 70, 71, 73, 75, 85]
# y1 = [91.83526, 89.63392, 89.29365, 88.546097, 94.23267, 94.661125, 96.470657, 95.491066]


# # Create the plot
# plt.figure(figsize=(10, 6))
# plt.plot(x, y1, marker='o', color='blue', markerfacecolor='blue', markeredgecolor='blue')


# # Add labels, legend, and title
# plt.xlabel('Training epochs')
# plt.xticks(x)
# plt.ylabel('Model Performace (w_rmse)')
# # plt.title('Three Series on the Same Plot')
# # plt.legend()
# plt.grid(True)

# # Save the figure
# # plt.savefig('epochs.png', dpi=300, bbox_inches='tight')

# # Optionally show the plot
# plt.show()
