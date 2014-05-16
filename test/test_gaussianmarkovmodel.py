import unittest
from six.moves import cPickle

import numpy as np
import scipy.optimize
from sklearn.mixture import gmm

from mixtape.tica import tICA
from mixtape.gaussianmarkovmodel import GaussianMarkovModel, gaussianMMFuncAndGrad


def gaussianMMFunc_np():
    def func(X_concat, X_breaks, mu, covars, lag_time, n_timescales, gamma):
        tica = tICA(lag_time=lag_time, gamma=gamma)
        chi_concat = np.exp(gmm._log_multivariate_normal_density_diag(
            X_concat, mu, covars))
        for i in range(len(X_breaks) - 1):
            tica.partial_fit(chi_concat[X_breaks[i] : X_breaks[i+1]])
        return tica.eigenvalues_[:n_timescales].sum()
    return func


class Test1(unittest.TestCase):
    def setUp(self):
        self.random = np.random.RandomState(0)
        self.n_samples, self.n_states, self.n_features = 10, 3, 2
        self.means = self.random.randn(self.n_states, self.n_features)
        self.covars = self.random.rand(self.n_states, self.n_features)
        self.X = self.random.randn(self.n_samples, self.n_features)
        self.X_breaks = [0, self.n_samples/2, self.n_samples]
        self.gamma = self.random.rand()
        self.gamma = 0.1

        self.f1 = gaussianMMFuncAndGrad()
        self.f2 = gaussianMMFunc_np()

    def test_a(self):
        # check that the values of the
        #  (1) numpy implementation based on calling the tica.py and
        #  (2) the Theano implementation
        # give the same result
        v1, g1, g2, _, _, _ = self.f1(self.X, self.X_breaks, self.means, self.covars, 1, 2, self.gamma)
        v2 = self.f2(self.X, self.X_breaks, self.means, self.covars, 1, 2, self.gamma)

        np.testing.assert_almost_equal(v1, v2)

    def test_b(self):
        # check that the gradients of the theano implementation against
        # finite difference
        def fg(x):
            means, covars = np.vsplit(x.reshape(self.n_states*2, self.n_features), 2)
            f, g1, g2, _, _, _ = self.f1(self.X, self.X_breaks, means, covars, 1, 2, self.gamma)
            print('f={}, g1={} g2={}'.format(f, g1, g2))
            return f, np.vstack((g1, g2)).reshape(2*self.n_states * self.n_features)

        x0 = self.random.rand(2*self.n_states*self.n_features)
        err = scipy.optimize.check_grad(lambda x: fg(x)[0], lambda x: fg(x)[1], x0)
        assert err < 1e-5


def test1():
    # make sure that the model is pickleable
    model = GaussianMarkovModel(n_components=2)
    cPickle.dumps(model)

    model.fit([np.random.RandomState(0).randn(100,1)])
    cPickle.dumps(model)
