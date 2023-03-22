import numpy as np
import util
import matplotlib as plt

def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    g = GDA()
    print(g.fit(x_train,y_train))

    x_e , y_e = util.load_dataset(valid_path,add_intercept=False)
    plot_path = save_path.replace('.txt','.png')
    util.plot(x_e,y_e,g.theta,plot_path)
    x_e = util.add_intercept(x_e)

    p_e = g.predict(x_e)
    yhat = p_e > 0.5
    print('GDA Accuracy: %.2f' % np.mean( (yhat == 1) == (y_e == 1)))
    np.savetxt(save_path, p_e)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True, theta0 = 0):
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
        self.theta0 = theta0

    def phi(self,x,y):

        sum = 0
        for i in range(y.shape[0]):

            if y[i]==1:

                sum+=1
            
        phi = sum/y.shape[0]
        return phi

    def mean(self,x,y,classi):

        a = np.zeros((x.shape[1]))
        k = 0

        for i in range(x.shape[0]):

            if y[i] == classi:

                a = np.add(a,x[i])
                k += 1

        return a/k

    def var(self,x,y):

        g = GDA()

        mew0 = g.mean(x,y,0)
        mew1 = g.mean(x,y,1)

        sig = np.zeros((x.shape[1],x.shape[1]))

        for i in range(y.shape[0]):

            if y[i]==0:
                sig = np.add(sig,np.outer((x[i]-mew0),(x[i]-mew0)))

            else:
                sig = np.add(sig,np.outer((x[i]-mew1),(x[i]-mew1)))

        sig = sig/y.shape[0]
        return sig



    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        
        g = GDA()
        phi = g.phi(x,y)
        mew0 = g.mean(x,y,0)
        mew1 = g.mean(x,y,1)
        sig = g.var(x,y)

        a = np.dot(mew0,np.dot(np.linalg.inv(sig),mew0))
        b = np.dot(mew1,np.dot(np.linalg.inv(sig),mew1))

        self.theta = np.zeros(x.shape[1]+1)

        g.phi(x,y)
        self.theta[0] = 0.5*(a-b) - np.log((1-phi)/(phi))
        self.theta[1:] = np.dot((mew1-mew0),np.linalg.inv(sig))

        return self.theta

    @staticmethod
    def _sigmoid(x):

        return 1/(1+np.exp(-x))

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """

        a = self._sigmoid(x.dot(self.theta))
        return a

        # *** START CODE HERE ***
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ps1/src/linearclass/ds1_train.csv',
         valid_path='ps1/src/linearclass/ds1_valid.csv',
         save_path='ps1/src/linearclass/gda1.txt')

    main(train_path='ps1/src/linearclass/ds2_train.csv',
         valid_path='ps1/src/linearclass/ds2_valid.csv',
         save_path='ps1/src/linearclass/gda2.txt')