import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define parameters of the normal distribution
mean = 0  # Mean (μ) of the normal distribution
std_dev = 1  # Standard deviation (σ) of the normal distribution

# Define the range of x values to plot
x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)

# Calculate the cumulative distribution function (CDF) values
cdf_values = norm.cdf(x, loc=mean, scale=std_dev)

# Plot the CDF
plt.plot(x, cdf_values, label=f'Normal Distribution CDF (mean={mean}, std_dev={std_dev})')

# Add labels and title
plt.xlabel('x')
plt.ylabel('Cumulative Probability')
plt.title('CDF of Normal Distribution')

# Add a legend
plt.legend()

# Show the plot
plt.show()