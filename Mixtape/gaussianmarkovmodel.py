"""Experimental
"""
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

# Mixtape is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Mixtape. If not, see <http://www.gnu.org/licenses/>.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function, division, absolute_import

import os
import time
import functools
import numpy as np
import scipy.optimize
import scipy.linalg
from sklearn import cluster
from sklearn.mixture import gmm
from sklearn.base import BaseEstimator
from sklearn.mixture import distribute_covar_matrix_to_match_covariance_type
from mixtape.tica import tICA

try:
    import theano
    from theano.printing import Print
    from theano import tensor as T
    from theano.sandbox import linalg
    linalg.eigvalsh  # make sure this exists
    imported_theano = True
except (ImportError, AttributeError):
    imported_theano = False


__all__ = ['GaussianMarkovModel']

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _memoized(obj):
    # Memoize decorator from
    # https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
    cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer


def _maximize_matfun(f_and_g, x0, method=None, bounds=None, tol=None, options=None):
    """Wrapper around scipy.optimize.minimize to maximize a function that
    takes a matrix as input"""
    assert x0.ndim == 2
    n, m = x0.shape
    iteration = [0]

    def minus_f_and_g(v):
        x = v.reshape(n, m)
        f, g = f_and_g(x)
        g = g.reshape(n*m)
        return -f, -g

    result = scipy.optimize.minimize(fun=minus_f_and_g, x0=x0.reshape(n*m),
        jac=True, tol=tol, method=method, options=options, bounds=bounds)
    return result


def _multivariate_normal_density_diag(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model

    This is a theano symbolic calculation
    """
    n_samples, n_dim = X.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + T.sum(T.log(covars), 1)
                  + T.sum((means ** 2) / covars, 1)
                  - 2 * T.dot(X, (means / covars).T)
                  + T.dot(X ** 2, (1.0 / covars).T))
    return T.exp(lpr)


@_memoized
def gaussianMMFuncAndGrad():
    assert imported_theano, 'The latest version of theano is required'

    X_concat = T.dmatrix('X')           # n_samples total x n_features
    X_breaks = T.ivector('X_breaks')    # indices of the breaks in X_concat
    mu = T.matrix('mu')                 # n_components x n_features
    covars = T.dmatrix('covars')        # n_components x n_features
    lag_time = T.scalar('lag_time', dtype='int64')
    n_timescales = T.scalar('n_timescales', dtype='int64')
    gamma = T.scalar('gamma', dtype='floatX')

    def _accumulate(index, SS, CC, MM, chi_concat, chi_breaks, lag_time):
        chi = chi_concat[chi_breaks[index]:chi_breaks[index+1], :]
        MM = MM + chi[lag_time:].sum(0) + chi[:-lag_time].sum(0)
        SS = SS + (T.dot(chi[:-lag_time].T, chi[:-lag_time])
                   + T.dot(chi[lag_time:].T, chi[lag_time:]))
        corrs = T.dot(chi[:-lag_time].T, chi[lag_time:])
        CC = CC + (corrs + corrs.T)
        return SS, CC, MM

    # chi has shape n_samples x n_components
    chi_concat = _multivariate_normal_density_diag(X_concat, mu, covars)
    # changing chi_concat to just X_concat recovers standard tICA. i.e:
    # chi_concat = X_concat + 0*mu.sum() + 0*covars.sum()
    n_components = chi_concat.shape[1]
    n_features = mu.shape[1]

    (SS, CC, MM), _ = theano.reduce(
        fn=_accumulate,
        outputs_info=[T.zeros((n_components, n_components)),
                      T.zeros((n_components, n_components)),
                      T.zeros((n_components,))],
        sequences=[T.arange(X_breaks.shape[0]-1)],
        non_sequences=[chi_concat, X_breaks, lag_time]
    )

    # S (overlap matrix) has shape n_components x n_components
    two_N = 2*(X_concat.shape[0] - lag_time * (X_breaks.shape[0] - 1))
    chimeans = (1.0 / two_N) * MM

    S = (1.0 / two_N) * SS - T.outer(chimeans, chimeans)
    rhs = S + (gamma / n_components) * (linalg.trace(S) * T.eye(n_components))

    # C (correlation matrix) has shape n_components x n_components
    C = (1.0 / two_N) * CC - T.outer(chimeans, chimeans)
    eigenvalues = linalg.eigvalsh(C, rhs)

    R = eigenvalues[-n_timescales:].sum()
    gradR = theano.gradient.grad(R, [mu, covars])

    f = theano.function(
        [X_concat, X_breaks, mu, covars, lag_time, n_timescales, gamma],
        [R, gradR[0], gradR[1], C, rhs, chimeans])

    return f


class GaussianMarkovModel(BaseEstimator):
    """
    Markov model with "soft" Gaussian states

    Parameters
    ----------
    n_components : int
        The number of basis functions in the model
    lag_time : int
        Delay time forward or backward in the input data. The time-lagged
        correlations is computed between datas X[t] and X[t+lag_time].
    n_timescales : int, default = n_components
        Number of eigenvalues to optimize. Defaults to all of them.
    gamma : nonnegative float, default=0.05
        Regularization strength. Positive `gamma` entails incrementing
        the sample covariance matrix by a constant times the identity,
        to ensure that it is positive definite. The exact form of the
        regularized sample covariance matrix is ::

            covariance + (gamma / n_features) * Tr(covariance) * Identity

        where :math:`Tr` is the trace operator.

    Other Parameters
    ----------------
    random_states : int, optional
        The generator used to initialize the means. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    opt_method : str
        Specify the optimization method to use. Should be one of ['L-BFGS-B',
        'TNC', 'SLSQP']. This string is passed directly to
        scipy.optimize.minimize.
    opt_tol : float
        Tolerance for termination of the optimizer. For detailed control,
        use solver-specific options.
    opt_options : dict
        A dictionary of solver options. All methods accept the following
        generic options:
            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.
        For method-specific options, see :func:`scipy.optimize.show_options()`.
        This value is passed directly to scipy.optimize.minimize.

    Attributes
    ----------
    means_ : array, shape (n_components, n_features)
        Mean parameters for each basis function
    covars_ : array, shape (n_components, n_features)
        Variances parameters for each basis function
    eigenvalues_ : array, shape=(n_timescales,)
        Generalized eigenvalues, in descending order.
    timescales_ : array, shape=(n_timescales,)
        The models relaxation timescales, in descending order.
    eigenvectors_ : array, shape=(n_components, n_timescales)
        Generalized eigenvectors.

    See Also
    --------
    scipy.optimize.minimize : used to optimize the model during fit()
    """

    def __init__(self, n_components=2, lag_time=1, n_timescales=None, gamma=0.05,
                 random_state=None, opt_method='BFGS', opt_tol=None,
                 opt_options=None):
        self.n_components = n_components
        self.lag_time = lag_time
        self.n_timescales = n_timescales
        self.gamma = gamma
        self.random_state = random_state
        self.opt_method = opt_method
        self.opt_tol = opt_tol
        self.opt_options = opt_options

    def _hotstart(self, X):
        """Hot-start means and covariances

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        means : array-like, shape=(n_components, n_features)
        """
        means = cluster.KMeans(
            n_clusters=self.n_components, n_init=1, n_jobs=1,
            random_state=self.random_state).fit(X).cluster_centers_

        cv = np.cov(X.T)
        if not cv.shape:
            cv.shape = (1, 1)
        covars = distribute_covar_matrix_to_match_covariance_type(
            cv, 'diag', self.n_components)
        covars[covars == 0] = 1e-5
        return means, covars

    def fit(self, sequences, y=None):
        """Estimate model parameters

        Parameters
        ----------
        sequences : list of array-like, each of shape [sequence_length, n_features]
            A list of multivariate timeseries. Each sequence may have
            a different length, but they all must have the same number
            of features.

        Returns
        -------
        self
        """
        # check sequences
        assert isinstance(sequences, list), 'sequences must be a list of arrays'
        X_breaks = np.cumsum([0,] + [len(s) for s in sequences],
                             dtype=np.int32)
        X_concat = np.concatenate(sequences)
        assert X_concat.ndim == 2

        n_features = X_concat.shape[1]

        # How many timesccales?
        n_timescales = self.n_timescales
        if n_timescales is None:
            n_timescales = self.n_components

        # Run hotstarting if necessary
        if hasattr(self, 'means_') and hasattr(self, 'covars_'):
            means0 = self.means_
            covars0 = self.covars_
        else:
            means0, covars0 = self._hotstart(X_concat)
            if hasattr(self, 'means_'):
               means0 = self.means_
            if hasattr(self, 'covars_'):
               covars0 = self.covars_

        # Get the thunk and make the function (objective, grad) callalable
        # to optimize
        thunk = gaussianMMFuncAndGrad()
        def f_and_g(v):
            mu, cov = np.vsplit(v, 2)
            f, gm, gc, _, _, _ = thunk(X_concat, X_breaks, mu, cov,
                self.lag_time, n_timescales, self.gamma)
            return f, np.vstack((gm, gc))

        # We don't want covars to go negative
        EPS = 1e-10
        bounds = ([(None, None) for _ in range(means0.size)] +
                  [(EPS, None) for _ in range(covars0.size)])

        # run the optimization and set the means_ and covars_ attributes
        result = _maximize_matfun(f_and_g, np.vstack((means0, covars0)), self.opt_method,
                                  bounds=bounds, tol=self.opt_tol, options=self.opt_options)
        self.means_, self.covars_ = np.vsplit(result.x.reshape(2*self.n_components, n_features), 2)

        # now we've got to get the eigenvectors too
        _, _, _, C, rhs, self.chimeans_ = thunk(X_concat, X_breaks,
            self.means_, self.covars_, self.lag_time, n_timescales, self.gamma)

        eigvals_range = (C.shape[0]-n_timescales, C.shape[0]-1)
        eigenvalues_, eigenvectors_ = scipy.linalg.eigh(C, rhs, eigvals=eigvals_range)
        order = np.argsort(-np.real(eigenvalues_))

        self.eigenvalues_ = eigenvalues_[order]
        self.eigenvectors_ = eigenvectors_[:, order]
        self.timescales = -self.lag_time / np.log(self.eigenvalues_)

        return self

    def transform(self, sequences):
        """Apply the dimensionality reduction on a collection of seqences

        Parameters
        ----------
        sequences: list of array-like, each of shape (n_samples_i, n_features)
            Training data, where n_samples_i in the number of samples
            in sequence i and n_features is the number of features.

        Returns
        -------
        sequence_new : list of array-like, each of shape (n_samples_i, n_components)
        """
        return [self.partial_transform(s) for s in sequences]

    def partial_transform(self, X):
        """Apply the dimensionality reduction on X

        Parameters
        ----------
        X :  array-like, shape=(n_samples, n_features)

        """
        chi = np.exp(gmm._log_multivariate_normal_density_diag(X, self.means_, self.covars_))
        return np.dot(chi - self.chimeans_, self.eigenvectors_)
