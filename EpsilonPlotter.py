import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
y = (7.0, 2.0, 0.0, 0.0, 0.0, 2.0, 7.0)

data_set = pd.DataFrame(x, y)

# Parameter range with a step size of 0.01
parameter_range = np.arange(-1, 1, 0.01)

# Initialize the 2D array to store epsilon sums
epsilon_sums = np.zeros((len(parameter_range), len(parameter_range)))

# Iterate over the range of b1 and b2
for b1_index, b1 in enumerate(parameter_range):
    for b2_index, b2 in enumerate(parameter_range):
        epsilon_sum = 0
        
        # Calculate the sum of errors (epsilon) for each (b1, b2) pair
        for item in range(len(x)):
            x_value = x[item]
            y_actual = y[item]
            y_estimate = b1 * x_value + b2 * x_value**2
            epsilon = abs(y_actual - y_estimate)
            epsilon_sum += epsilon
        
        # Store the sum of epsilon for the current (b1, b2) pair
        epsilon_sums[b1_index, b2_index] = epsilon_sum

# Find the indices with the minimum epsilon sum
min_error = np.min(epsilon_sums)
min_indices = np.unravel_index(np.argmin(epsilon_sums), epsilon_sums.shape)
best_b1 = parameter_range[min_indices[0]]
best_b2 = parameter_range[min_indices[1]]
print(f"The best parameters (b1, b2) pair with the least sum of errors is ({best_b1}, {best_b2}) with a sum of errors (min epsilon) of {min_error}.")

# Create a 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for b1 and b2 values
b1_grid, b2_grid = np.meshgrid(parameter_range, parameter_range)

# Plot the surface
surface_plot = ax.plot_surface(b1_grid, b2_grid, epsilon_sums, cmap='viridis')

# Add a colorbar to represent the range of epsilon sums
colorbar = fig.colorbar(surface_plot, ax=ax, shrink=0.5, aspect=5)
colorbar.set_label('Sum of Epsilons')

# Set labels and title for the plot
ax.set_xlabel('b1 [linear scale]')
ax.set_ylabel('b2 [linear scale]')
ax.set_zlabel('Sum of Epsilons')
ax.set_title('Plot parameters b1 & b2 to identify minimum epsilon')

# Show the plot
plt.show()
