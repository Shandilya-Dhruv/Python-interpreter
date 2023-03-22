# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
from matplotlib import pyplot as plt


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape

    probs = 1. / (1 + np.exp(-X.dot(theta)))
    grad = (Y - probs).dot(X)

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = 0.1

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta + learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-5:
            print('Converged in %d iterations' % i)
            break

    print(theta)
    print(grad)
    return


def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('ps2/src/stability/ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('ps2/src/stability/ds1_b.csv', add_intercept=True)
    for i in range(Yb.shape[0]):

        if Yb[i] == 0:
            plt.scatter(Xb[i][1],Xb[i][2],c='blue')

        else:
            plt.scatter(Xb[i][1],Xb[i][2],c='red')

    plt.savefig('hello1.png')
    plt.clf()


if __name__ == '__main__':
    main()
