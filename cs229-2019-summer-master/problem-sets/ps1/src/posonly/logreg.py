import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_validation, y_validation = util.load_dataset(valid_path, add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_train,y_train)
    
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_eval, y_eval, clf.theta, plot_path)

    # Use np.savetxt to save predictions on eval set to save_path
    p_eval = clf.predict(x_eval)
    yhat = p_eval > 0.5
    print('LR Accuracy: %.2f' % np.mean( (yhat == 1) == (y_eval == 1)))
    np.savetxt(save_path, p_eval)
    
    # d = clf.predict(x_validation)

    # for i in range(len(d)):
    #     print(d[i])
    #     print(y_validation[i])
    
    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=np.array([0,0,0]), verbose=True):
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

    def phi(self, a):
        
        b = 1/(1+np.exp(-1*a))

        return b

    def first_der(self, theta, x, y):
        
        dj = np.zeros(theta.size)
        o = LogisticRegression()

        for j in range(theta.size):
            a = 0
            for i in range(x.shape[0]):
                a += ( y[i] - o.phi(np.dot(theta,x[i])) )*(x[i,j])
            dj[j] = a
        return dj

    def hes(self, theta, x):

        o = LogisticRegression()

        hes = np.zeros((x.shape[1],x.shape[1]))

        for i in range(x.shape[0]):
            hes = np.add(hes,np.outer(o.phi(np.dot(theta,x[i]))*np.transpose(x[i]),(1-o.phi(np.dot(theta,x[i])))*x[i]))

        return -1*hes

    def grad_asc(self, x, y):

        o = LogisticRegression()

        theta = self.theta

        first_der = o.first_der(theta,x,y)
        alpha = 0.001

        while np.linalg.norm(alpha*first_der) > self.eps:
            theta = theta + alpha*first_der
            first_der = o.first_der(theta,x,y)
        
        return theta

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """

        o = LogisticRegression()

        theta = self.theta

        H_inv = np.linalg.inv(o.hes(theta,x))
        der = o.first_der(theta,x,y)

        i = 0

        while np.linalg.norm(np.dot(H_inv,der))>self.eps:

            theta = theta - np.dot(H_inv,der)
            H_inv = np.linalg.inv(o.hes(theta,x))
            der = o.first_der(theta,x,y)

        self.theta = theta

        return theta

        

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        
        theta = self.theta
        o = LogisticRegression()

        d = o.phi(np.dot(x,self.theta))
        
        return d

if __name__ == '__main__':
    main(train_path='ps1/src/linearclass/ds1_train.csv',
         valid_path='ps1/src/linearclass/ds1_valid.csv',
         save_path='ps1/src/linearclass/logreg_pred_1.txt')

    main(train_path='ps1/src/linearclass/ds2_train.csv',
         valid_path='ps1/src/linearclass/ds2_valid.csv',
         save_path='ps1/src/linearclass/logreg_pred_2.txt')
