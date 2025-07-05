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
