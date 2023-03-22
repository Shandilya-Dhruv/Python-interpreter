import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    poi = PoissonRegression()

    poi.fit(x_train,y_train)

    x_eval,y_eval = util.load_dataset(eval_path, add_intercept= True)
    p_eval = poi.predict(x_eval)
    np.savetxt(save_path, p_eval)
    plt.figure()
    plt.scatter(y_eval,p_eval,alpha=0.4,c='red',label='Ground Truth vs Predicted')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.legend()
    plt.savefig('poisson_valid.png')

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to save_path
    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def der(self,x,y,theta):

        d = np.zeros(x.shape[1])

        for i in range(d.shape[0]):

            var = 0

            for j in range(x.shape[0]):

                b = np.exp(np.dot(theta,x[j]))

                var += x[j][i]*(y[j]-b)

            d[i] = var
        
        return d



    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        
        o = PoissonRegression()

        theta = np.zeros(x.shape[1])
        d = o.der(x,y,theta)

        while self.step_size*np.linalg.norm(d)>self.eps :

            theta = theta + self.step_size*d
            d = o.der(x,y,theta)

        self.theta = theta
        return theta


    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***

        pred = np.zeros(x.shape[0])
        
        for i in range(x.shape[0]):

            pred[i] = np.exp(np.dot(self.theta,x[i]))

        return pred

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='ps1/src/poisson/train.csv',
        eval_path='ps1/src/poisson/valid.csv',
        save_path='ps1/src/poisson/poisson_pred.txt')
