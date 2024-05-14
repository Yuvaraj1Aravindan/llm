import numpy as np
import matplotlib.pyplot as plt

def plot_normal_distribution(mu, sigma, ax, label=None, color=None):
    """
    Plots the normal distribution for a given mean (mu) and standard deviation (sigma).
    
    Parameters:
        mu (float): The mean of the distribution.
        sigma (float): The standard deviation of the distribution.
        ax (matplotlib Axes): The Axes object to plot on.
        label (str): The label for the plot (used in the legend).
        color (str): The color for the plot.
    """
    # Define x values for the range from mu - 5*sigma to mu + 5*sigma
    x_min = mu - 5 * sigma
    x_max = mu + 5 * sigma
    x = np.linspace(x_min, x_max, 1000)

    # Calculate the normal distribution
    normalization_factor = 1 / (sigma * np.sqrt(2 * np.pi))
    y = normalization_factor * np.exp(-((x - mu)**2) / (2 * sigma**2))

    # Plot the normal distribution on the given axes
    ax.plot(x, y, label=label, color=color)

# Initialize a figure and axes
fig, ax = plt.subplots()

# Define a custom color palette without red and green
custom_colors = ['blue', 'black', 'orange', 'red', 'brown', 'magenta']

# Plot different normal distributions
# Case 1: Different standard deviations, same mean
mu = 0
sigmas = [0.5, 1, 2]
for idx, sigma in enumerate(sigmas):
    color = custom_colors[idx % len(custom_colors)]
    plot_normal_distribution(mu, sigma, ax, label=f'μ = {mu}, σ = {sigma}', color=color)

# Case 2: Different means, same standard deviation
sigma = 1
means = [1, 2, 3]
for idx, mu in enumerate(means):
    color = custom_colors[(idx + len(sigmas)) % len(custom_colors)]
    plot_normal_distribution(mu, sigma, ax, label=f'μ = {mu}, σ = {sigma}', color=color)

# Add title, labels, and legend
ax.set_title('Normal Distributions')
ax.set_xlabel('x [scale = linear]\n\n Plots the normal distribution for a given mean (mu) and standard deviation (sigma)')
ax.set_ylabel('y = f(x)')
ax.legend()
#plt.figtext(0.5, 0.05, 'description')
# Display the plot
plt.show()