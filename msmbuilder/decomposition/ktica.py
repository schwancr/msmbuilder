
import numpy as np
import scipy.linalg
from mdtraj import io
import pickle
from ..base import BaseEstimator
from sklearn.base import TransformerMixin

class ktICA(BaseEstimator, TransformerMixin):
    """ 
    Time-structure Indpendent Componenent Analysis (tICA) using the kernel
    trick. 

    The kernel trick allows one to extend a linear method (e.g. tICA) to
    include non-linear solutions. 

    <Add some exposition here>

    Parameters
    ----------
    kernel : str or callable
        The kernel function to define similarities in the feature space.
        It must be one of:
            - 'linear' : linear kernel (dot product in the input space)
            - 'poly' : polynomial kernel
            - 'rbf' : radial basis function 
            - 'sigmoid' :
            - 'precomputed' : the precomputed gram matrix will be passed
                              to ktICA.fit().
            - callable : function that takes two datasets and returns
                         a matrix of similarities
    degree : int, optional
        Degree of the polynomial kernel. This is only used if kernel
        is 'poly'.
    gamma : float, optional
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. If gamma == 0.0,
        then will use 1 / n_features.
    coef0 : float, optional
        Independent term in 'poly' and 'sigmoid'.
    lag_time : int, optional
        Lag time to define the offset correlation matrix.
    stride : int, optional
        Only sample pairs of points from the data according to this stride
    eta : float, optional
        Regularization strength for solving the ktICA problem
    n_components : int, optional
        Number of components to project onto in the `transform` method
        and to use when scoring.

    Attributes
    ----------
    components_ : array-like, shape (n_components, n_datapoints)
        Components with maximum autocorrelation; n_datapoints is equal
        to twice the number of pairs of points used to fit the model.
    eigenvalues_ : array-like, shape (n_datapoints,)
        Eigenvalues of the tICA generalized eigenproblem, in decreasing
        order.
    eigenvectors_ : array-like, shape (n_datapoints, n_datapoints)
        ALLT Eigenvectors of the tICA generalized eigenproblem. The vectors
        give a set of "directions" through configuration space along
        which the system relaxes towards equilibrium. Each eigenvector
        is associated with characteristic timescale
        :math:`- \frac{lag_time}{ln \lambda_i}, where :math:`lambda_i` is
        the corresponding eigenvector. 
        The eigenvectors are stored in columns and normalized to have
        unit variance according to the training data.
    uncentered_gram_matrix_ : array-like, shape (n_datapoints, n_datapoints)
        The uncentered gram matrix of similarities.
    n_datapoints_ : int
        Total number of data points fit by the model. This is equal to
        two times the number of pairs sampled during fitting.
    timescales_ : array-like, shape (n_components,)
        The implied timescales of the tICA model, given by
        - lag_time / log(eigenvalues)

    Notes
    -----
    There are many ways to approximate the kernel solution to the tICA
    problem that are not nearly as computationally intense as this 
    method. Checkout sklearn.kernel_approximation for details.
    """
    _available_kernels = ['rbf', 'sigmoid', 'linear', 'poly'] 
    
    def __init__(self, kernel='rbf', degree=3, gamma=1.0, lag_time=1, 
                 stride=1, n_components=1, eta=1.0):

        if not kernel in _available_kernels:
            if kernel != 'precomputed':
                if not callable(kernel):
                    raise ValueError("kernel must be one of %s or 'precomputed' or a callable function" % str(_available_kernels))

        self.kernel = kernel

        self.degree = int(degree)
        self.gamma = float(gamma)
        self.coef0 = float(coef0)
        self.eta = float(eta)


        self.lag_time = int(lag_time)
        self.stride = int(stride)
        self.n_components = int(n_components)

    @property
    def components_(self):
        return self.eigenvectors_[:, self.n_components].T


    @property
    def timescales_(self):
        return - self.lag_time / np.log(self.eigenvalues_[:self.n_components])


    @property
    def _kernel_params(self):
        return {'degree' : self.degree, 'gamma' : self.gamma, 
                'coef' : self.coef0}


    def fit(self, X):
        r"""
        Fit the model to a given timeseries
        
        Parameters
        ----------
        X : array-like, shape = [n_sequences, n_samples, n_features]
            if X_dt is None, then this is a time series and pairs of 
            points will be sampled from it. Otherwise, these are the
            initial points to the corresponding points in X_dt
            
            If the kernel you specified is 'precomputed', then X should
            be the UNCENTERED gram matrix of inner products such that:
        """

        if self.kernel == 'precomputed':
            if X.shape[0] != X.shape[1]:
                raise ValueError("X is supposed to be square for kernel='precomputed'")

            self.uncentered_gram_matrix_ = X
            self.__Xfit = None
            # this will throw errors if we try to access this later
            self.n_datapoints_ = X.shape[0]

        else:
            # sequences holding the subsampled points
            X_0 = []
            X_t = []

            for seq in X:
                seq_t = seq[self.lag_time::stride]
                seq_0 = seq[::stride][:len(seq_t)]
                X_0.append(seq_0)
                X_t.append(seq_t)

            self.__Xfit = np.concatenate([np.concatenate(X_0), np.concatenate(X_t)])

            self.n_datapoints_ = len(self.__Xfit)

            self.uncentered_gram_matrix_ = pairwise_kernels(self.__Xfit, metric=self.kernel,
                                                            n_jobs=-1, filter_params=True,
                                                            kwds=self._kernel_params)           

        # just make sure it's actually symmetric
        self.uncentered_gram_matrix = (self.uncentered_gram_matrix_ + self.uncentered_gram_matrix_.T) * 0.5

        # matrix used when centering
        oneN = np.ones(self.n_datapoints_) / float(self.n_datapoints_)
        oneN.reshape((-1, 1))

        self.gram_matrix_ = self.uncentered_gram_matrix_ \
                            - oneN.T.dot(self.uncentered_gram_matrix_) \
                            - self.uncentered_gram_matrix_.dot(oneN) \
                            + oneN.T.dot(self.uncentered_gram_matrix_.dot(oneN))

        # again just make sure that there's no rounding issues
        self.gram_matrix_ = (self.gram_matrix_ + self.gram_matrix_.T) * 0.5

        n_pairs = self.n_datapoints / 2
        R = np.zeros(self.n_datapoints_)
        R[:n_pairs, n_pairs:] = np.eye(n_pairs)
        R[n_pairs:, :n_pairs] = np.eye(n_pairs)

        KK = self.gram_matrix_.dot(self.gram_matrix_)

        lhs = self.gram_matrix_.dot(R).dot(self.gram_matrix_)
        rhs = KK + self.eta * np.eye(self.n_datapoints_)

        vals, vecs = scipy.linalg.eigh(lhs, b=rhs)

        dec_ind = np.argsort(vals)[::-1]
        
        self.eigenvalues_ = vals[dec_ind]
        vecs = vecs[:, dec_ind]

        # now, normalize the eigenvectors to have unit variance
        vKK = vecs.T.dot(KK)
        # not sure if I should compute the variance based on
        # the regularization strength or not :/
        vec_vars = np.sum(vKK * vecs.T, axis=1) / self.n_datapoints_
        self.eigenvectors_ = vecs / np.sqrt(vec_vars)


    def transform(self, X):
        """
        project a point onto the top `n_components` ktICs

        Parameters
        ----------
        X : np.ndarray, shape = [n_points, n_features]
            Data to project onto eigenvector. If kernel == 'precomputed'
            Then this must be a kernel matrix, where:
                X[i, j] = kernel(Xfit[i], Xnew[j]) 
            where Xfit are all of the points used to fit the model.

        Returns
        -------
        Xnew : np.ndarray, shape = [n_points, n_components]
            projected value of each point in the trajectory
        """

        Ku = pairwise_kernels(X, Y=self.__Xfit, metric=self.kernel,
                              filter_params=True, n_jobs=-1, 
                              **self._kernel_params)

        N = Ku.shape[0]
        oneN = np.ones((N, 1)) / float(N)

        K = Ku - self.uncentered_gram_matrix_.dot(oneN) \
            - oneN.T.dot(Ku) \
            + oneN.T.dot(self.uncentered_gram_matrix_.dot(oneN))

        Xnew = K.T.dot(self.components_.T)
        
        return Xnew

    def score():
        raise NotImplementedError("I need to compute the GMRQ here.")
