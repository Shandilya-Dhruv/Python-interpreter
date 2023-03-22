from cmath import sin
import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        y_new = np.expand_dims(y, axis=1)
        self.theta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y_new))
        return self.theta

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        a = np.zeros([X.shape[0],k+1])
        for i in range(X.shape[0]):

            for j in range(k+1):

                a[i][j] = pow(X[i][1],j)

        return a

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        a = np.zeros([X.shape[0],k+2])
        for i in range(X.shape[0]):

            for j in range(k+1):

                a[i][j] = pow(X[i][1],j)

            a[i][k+1] = np.sin(X[i][1])

        return a

    def predict(self, X, theta):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """

        return np.dot(X,theta)


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    # plt.figure()
    # plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        o = LinearModel()
        
        if sine:
            x = o.create_sin(k,train_x)

        else:
            x = o.create_poly(k,train_x)

        theta = o.fit(x,train_y)

        if sine:
            poly_plot_x = o.create_sin(k,plot_x)

        else:
            poly_plot_x = o.create_poly(k,plot_x)

        plot_y = o.predict(poly_plot_x,theta)

        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)
        plt.scatter(train_x[:,1],train_y)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    run_exp(train_path, False, [3], 'large-poly3.png')
    run_exp(train_path, False, [1, 2, 3, 5, 10, 20], 'large-poly.png')
    run_exp(train_path, True, [1, 2, 3, 5, 10, 20], 'large-sine.png')
    run_exp(small_path, True, [1, 2, 3, 5, 10, 20], 'small-sine.png')
    run_exp(small_path, False, [1, 2, 3, 5, 10, 20], 'small-poly.png')

if __name__ == '__main__':
    main(train_path='ps1/src/featuremaps/train.csv',
        small_path='ps1/src/featuremaps/small.csv',
        eval_path='ps1/src/featuremaps/test.csv')
