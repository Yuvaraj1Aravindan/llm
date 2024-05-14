import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple, Tuple, List, Optional

# Define a NamedTuple for a range of values
class SampleRange(NamedTuple):
    start: float
    stop: float
    count: int

# Define noise function
def generateNoise(mean: float, sigma: float) -> float:
    """Generates noise from a normal distribution with a given mean and standard deviation."""
    return np.random.normal(mean, sigma)

# Function to generate x and y values with noise
def generateData(xRange: SampleRange, noiseMean: float, noiseSigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generates x and y values within a given range with added noise."""
    xValues = np.linspace(xRange.start, xRange.stop, xRange.count)
    noise = generateNoise(noiseMean, noiseSigma)
    yValues = (2 * xValues) - 3 + noise
    return xValues, yValues

# Calculate x/feature matrix for the closed-form solution
def createFeatureMatrixForErrorSurface(x: np.ndarray, degree: int) -> np.ndarray:
    """Creates the x matrix for the closed-form solution."""
    return np.array([[xi ** j for j in range(degree + 1)] for xi in x])

# Function to calculate beta values using closed-form solution
def calculateClosedFormBeta(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Calculates beta values using the closed-form solution for a polynomial of a given degree."""
    xMatrix = createFeatureMatrixForErrorSurface(x, degree)
    xTranspose = xMatrix.T
    beta = np.linalg.inv(xTranspose @ xMatrix) @ (xTranspose @ y)
    return beta

# Function to calculate the sum of squared errors
def calculateError(x: np.ndarray, y: np.ndarray, beta0: float, beta1: float) -> float:
    """Calculates the sum of squared errors."""
    predictions = beta0 + beta1 * x
    errors = y - predictions
    return np.sum(errors ** 2)

# Function to calculate the gradients for beta0 and beta1
def calculateGradients(x: np.ndarray, y: np.ndarray, beta0: float, beta1: float) -> Tuple[float, float]:
    """Calculates the gradients for beta0 and beta1."""
    predictions = beta0 + beta1 * x
    gradBeta0 = -2 * np.mean(y - predictions)
    gradBeta1 = -2 * np.mean(x * (y - predictions))
    return gradBeta0, gradBeta1

# Function to perform one iteration of gradient descent
def performGradientDescentIteration(x: np.ndarray, y: np.ndarray, beta0: float, beta1: float, learningRate: float) -> Tuple[float, float, float]:
    """Performs one iteration of gradient descent."""
    gradBeta0, gradBeta1 = calculateGradients(x, y, beta0, beta1)
    beta0 -= learningRate * gradBeta0
    beta1 -= learningRate * gradBeta1
    error = calculateError(x, y, beta0, beta1)
    return beta0, beta1, error

# Function to find beta values using gradient descent
def findBetaGradientDescent(x: np.ndarray, y: np.ndarray, learningRate: float, maxIterations: int = 10000, tol: float = 1e-6) -> Optional[Tuple[float, float, List[float]]]:
    """Finds beta values using the gradient descent optimization algorithm."""
    beta0, beta1 = np.random.normal(0, 1), np.random.normal(0, 1)
    previousError = calculateError(x, y, beta0, beta1)
    errors = [previousError]
    beta0_values = [beta0]
    beta1_values = [beta1]

    epochCount = 0

    for _ in range(maxIterations):
        beta0, beta1, error = performGradientDescentIteration(x, y, beta0, beta1, learningRate)
        error_difference = abs(errors[-1] - error)
        if error_difference < tol or error < 0.0001:
            if error_difference < tol:
                print(f"Terminating: Change in error ({error_difference:.6f}) is below tolerance ({tol:.6f}) at epoch=({epochCount}) and learningRate = ({learningRate}).")
            else:
                print(f"Terminating: Error ({error:.6f}) is below 0.0001 at epoch=({epochCount}) and learningRate = ({learningRate}).")
            break
        errors.append(error)
        beta0_values.append(beta0)
        beta1_values.append(beta1)
        epochCount+=1
        
    return beta0, beta1, errors, beta0_values, beta1_values

# Plot the data, closed-form solution, and gradient descent solution
def plotClosedFormSolution(xValues: np.ndarray, yValues: np.ndarray, closedFormBetas: np.ndarray) -> None:
    """Plots the original data points and the closed-form solution."""
    plt.figure(figsize=(8, 6))
    plt.scatter(xValues, yValues, color='gray', label='Data', alpha=0.6)
    yClosedForm = closedFormBetas[0] + closedFormBetas[1] * xValues
    plt.plot(xValues, yClosedForm, color='red', label='Closed-Form Solution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Closed-Form Solution')
    plt.legend()
    plt.show()

# Plot the data and gradient descent solution
def plotGradientDescentSolution(xValues: np.ndarray, yValues: np.ndarray, gradientDescentBetas: Tuple[float, float], learningRate: float) -> None:
    """Plots the original data points and the gradient descent solution."""
    plt.figure(figsize=(8, 6))
    plt.scatter(xValues, yValues, color='gray', label='Data', alpha=0.6)
    yGradientDescent = gradientDescentBetas[0] + gradientDescentBetas[1] * xValues
    plt.plot(xValues, yGradientDescent, color='blue', label='Gradient Descent Solution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"Gradient Descent Solution for learningRate ({learningRate})")
    plt.legend()
    plt.show()

# Plot the data, closed-form solution, and gradient descent solution
def plotSolutions(xValues: np.ndarray, yValues: np.ndarray, closedFormBetas: np.ndarray, gradientDescentBetas: Tuple[float, float], errors: List[float], beta0_values: List[float], beta1_values: List[float], learningRate: float) -> None:
    # Plot error over epochs for gradient descent
    plt.figure(figsize=(10, 6))
    iterations = range(len(errors))
    plt.plot(iterations, errors, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title(f"Error over Epochs for Gradient Descent for learningRate ({learningRate})")
    plt.show()

    # Plot beta0 and beta1 over epochs for gradient descent
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    iterations = range(len(beta0_values))
    plt.plot(iterations, beta0_values, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Beta0')
    plt.title(f"Beta0 over Epochs for Gradient Descent for learningRate ({learningRate})")

    plt.subplot(2, 1, 2)
    iterations = range(len(beta1_values))
    plt.plot(iterations, beta1_values, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Beta1')
    plt.title(f"Beta1 over Epochs for Gradient Descent for learningRate ({learningRate})")

    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Initialize the sample range and generate data
    xRange = SampleRange(start=-5, stop=5, count=100)
    xValues, yValues = generateData(xRange, noiseMean=0, noiseSigma=5)

    # Calculate beta values using closed-form solution
    closedFormBetas = calculateClosedFormBeta(xValues, yValues, degree=1)
    print(f"closedFormBetas: ({closedFormBetas}).")

    # Plot the closed-form solution
    plotClosedFormSolution(xValues, yValues, closedFormBetas)

    # Define a list of different learning rates
    etas = [0.001, .01, .1]

    # Iterate over each learning rate
    for eta in etas:
        # Calculate beta values using gradient descent
        gradientDescentResult = findBetaGradientDescent(xValues, yValues, learningRate=eta)

        if gradientDescentResult:
            gradientDescentBetas, errors, beta0_values, beta1_values = gradientDescentResult[0:2], gradientDescentResult[2], gradientDescentResult[3], gradientDescentResult[4]
            print(f"gradientDescentBetas for eta={eta}: ({gradientDescentBetas}).")
            # Plot the gradient descent solution
            plotGradientDescentSolution(xValues, yValues, gradientDescentBetas, eta)

            # Plot the data, closed-form solution, and gradient descent solution
            plotSolutions(xValues, yValues, closedFormBetas, gradientDescentBetas, errors, beta0_values, beta1_values, eta)

# Run the main function when the script is executed directly
if __name__ == "__main__":
    main()