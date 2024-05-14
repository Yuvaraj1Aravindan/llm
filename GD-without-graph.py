from __future__ import annotations

# Importing the numerical Python library
import numpy as np

# Importing matplotlib for plotting the graph
import matplotlib.pyplot as plt


# Function to generate a dataset based on the input x
def datasetGenerator(x: int) -> int:
    return ((2 * x) - 3) + np.random.normal(0, 5)


# mean squared error calculation for comparing the actual y with the prediction
def meanSquaredError(y: float, y_pred: float) -> float:
    # summation of squares of all the y with y predictions divided by the length of y
    return sum((y - y_pred) ** 2 / len(y))


def gradientDescent(
    X: list[float],
    Y: list[float],
    Y_pred: list[float],
    B0: float,
    B1: float,
    learningRate: float,
):
    B0 -= learningRate * np.mean(-2 * (Y - Y_pred))
    B1 -= learningRate * np.mean(2 * (Y - Y_pred) * (-X))
    return B0, B1


def betaCalculation(X: list[float], Y: list[float], n: int) -> int:
    # numpy array of X power 1 to 5
    Xtrans = [np.power(X, i) for i in range(n + 1)]

    # actual X values in form of X tranpose's transpose
    Xnew = np.transpose(Xtrans)

    # dot product of X transpose with X
    XTX = np.matmul(Xtrans, Xnew)

    # inverse of dot product of X Transpose with X
    XTXm1 = np.linalg.inv(XTX)

    # dot product of inverse of dot product of X transpose and X with X tranpose
    XTXinvintoXT = np.matmul(XTXm1, Xtrans)

    # Dot product of dot product of inverse of dot product of X transpose and X with X tranpose with Y
    Beta = np.matmul(XTXinvintoXT, Y)

    # array of beta elements
    return Beta


# gets executed when the file is executed
def main():
    # having a numpy array of range from -5 to 5 with 10000
    X = np.linspace(-5, 5, 10000)

    # generating y values with the
    Y = datasetGenerator(X)

    # calculating the B0 and B1 with normal random
    B0 = np.random.normal(0, 1)
    B1 = np.random.normal(0, 1)
    # putting the B0 and B1 in a list
    B = [[B0, B1]]
    flag = True
    Yvals = Y
    epsarr = []
    epochsarr = []
    epochs = 0
    while flag:
        # prediction of the model for previous B0 and B1
        Y_pred = B0 + B1 * X

        # checking if the code converged or not
        if meanSquaredError(Yvals, Y_pred) <= 1e-6:
            # if it converges changing the flag
            flag = False

        # finding the error of the mean squared error
        eps = meanSquaredError(Y, Y_pred)

        # calculating new B0 and B1 using gradient descent
        B0, B1 = gradientDescent(X, Y, Y_pred, B0, B1, learningRate=0.01)
        B.append([B0, B1])
        Yvals = Y_pred
        # adding the epoch value to see how it converges
        epochs += 1
        epsarr.append(eps)
        epochsarr.append(epochs)

    # printing the Beta in closedform and closedform solution
    print(B[-1])
    print(betaCalculation(X, Y, 1))

    # printing the epoch value to see how it converges
    print(epochs)

if __name__ == "__main__":
    main()