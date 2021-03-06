from __future__ import print_function
import time
import sys
import numpy as np
import scipy.linalg
from numpy.testing.decorators import skipif
from scipy.optimize import check_grad, approx_fprime
try:
    import numdifftools as nd
except ImportError:
    print('Missig test dependency', file=sys.stderr)
    print('  pip installl numdifftools', file=sys.stderr)
    raise

from msmbuilder.msm import _ratematrix
from msmbuilder.msm import ContinuousTimeMSM, MarkovStateModel
from msmbuilder.example_datasets import MullerPotential
from msmbuilder.example_datasets import load_doublewell
from msmbuilder.cluster import NDGrid
random = np.random.RandomState(0)


def example_theta(n):
    return random.uniform(0, 1, size=int(n*(n-1)/2 + n))


def test_build_ratemat_1():
    # test build_ratemat
    n = 4
    theta = example_theta(n)
    K = np.zeros((n, n))
    _ratematrix.build_ratemat(theta, n, K)

    # diagonal entries are negative
    assert np.all(np.diag(K) < 0)
    # off diagonal entries are non-negative
    assert np.all(np.extract(1-np.eye(n), K) >= 0)
    # row-sums are 0
    np.testing.assert_array_almost_equal(np.sum(K, axis=1), 0)


def test_dK_dtheta_1():
    # test function `dK_dtheta_A` against the numerical gradient of
    # `build_ratemat
    n = 4
    theta = example_theta(n)

    def func(x, i, j):
        # (i,j) entry of the rate matrix
        K = np.zeros((n, n))
        _ratematrix.build_ratemat(x, n, K)
        return K[i, j]

    def grad(x, i, j):
        # gradient of the (i,j) entry of the rate matrix w.r.t. theta
        dKu = np.zeros((n, n))
        g = np.zeros(len(x))
        for u in range(len(x)):
            _ratematrix.dK_dtheta_A(x, n, u, None, dKu)
            g[u] = dKu[i, j]
        return g

    for i in range(n):
        for j in range(n):
            assert check_grad(func, grad, theta, i, j) < 1e-7


def test_dK_dtheta_2():
    # test function `dK_dtheta_A` to make sure that the part that hadamards
    # the matrix against A is correct.
    n = 4
    A = random.randn(4, 4)
    theta = example_theta(n)

    for u in range(len(theta)):
        dKu = np.zeros((n, n))
        _ratematrix.dK_dtheta_A(theta, n, u, None, dKu)
        value1 = (dKu * A).sum()

        dKu = np.zeros((n, n))
        value2 = _ratematrix.dK_dtheta_A(theta, n, u, A, dKu)
        value3 = _ratematrix.dK_dtheta_A(theta, n, u, A)

        np.testing.assert_approx_equal(value1, value2)
        np.testing.assert_approx_equal(value1, value3)


def test_grad_logl_1():
    # test the gradient of the `loglikelihood` against a numerical gradient
    n = 4
    t = 1
    C = random.randint(10, size=(n, n)).astype(float)
    theta0 = example_theta(n)

    def func(theta):
        return _ratematrix.loglikelihood(theta, C, t=t)[0]

    def grad(theta):
        return _ratematrix.loglikelihood(theta, C, t=t)[1]

    assert check_grad(func, grad, theta0) < 1e-4


def test_dw_1():
    # test the gradient of the eigenvalues of K
    n = 5
    t = 1.0
    theta0 = example_theta(n)

    def grad(theta, i):
        # gradient of the ith eigenvalue of K with respect to theta
        K = np.zeros((n, n))
        _ratematrix.build_ratemat(theta, n, K)
        w, V = scipy.linalg.eig(K)
        order = np.argsort(np.real(w))

        V = np.real(np.ascontiguousarray(V[:, order]))
        U = np.ascontiguousarray(scipy.linalg.inv(V).T)

        g = np.zeros(len(theta))

        for u in range(len(theta)):
            dKu = np.zeros((n, n))
            _ratematrix.dK_dtheta_A(theta, n, u, None, dKu)
            out = np.zeros(n)
            temp = np.zeros(n)
            _ratematrix.dw_du(dKu, U, V, n, temp, out)
            g[u] = out[i]
        return g

    def func(theta, i):
        # ith eigenvalue of K
        K = np.zeros((n, n))
        _ratematrix.build_ratemat(theta, n, K)
        w = np.real(scipy.linalg.eigvals(K))
        w = np.sort(w)
        return w[i]

    for i in range(n):
        g1 = approx_fprime(theta0, func, 1e-7, i)
        g2 = grad(theta0, i)
        assert np.linalg.norm(g1-g2) < 2e-6


def test_hessian_1():
    n = 3
    seqs = [[1,1,1,1,1,2,2,2,2,1,1,1,1,3,3,3,3,3,2,2,2,2,2,2,1,1,1,1,2,3,3,3,3]]

    model = ContinuousTimeMSM().fit(seqs)
    theta = model.theta_
    C = model.countsmat_

    hessian1 = _ratematrix.hessian(theta, C)
    Hfun = nd.Jacobian(lambda x: _ratematrix.loglikelihood(x, C)[1])
    hessian2 = Hfun(theta)

    # not sure what the cutoff here should be (see plot_test_hessian)
    assert np.linalg.norm(hessian1-hessian2) < 1e-6

    print(_ratematrix.sigma_pi(-scipy.linalg.pinv(hessian1), theta, n))


# @skipif(True, 'Zeros in the count matrix -> known failure')
def test_hessian_2():
    n = 3
    seqs = [[1,1,1,1,1,2,2,2,2,1,1,1,1,2,3,3,3,3,3,2,2,2,2,2,2,1,1,1,1,2,3,3,3,3]]

    model = ContinuousTimeMSM().fit(seqs)
    print(model.timescales_)
    print(model.uncertainty_timescales())
    theta = model.theta_
    C = model.countsmat_
    print(C)

    C_flat = (C+C.T)[np.triu_indices_from(C, k=1)]
    print(C_flat)
    print('theta', theta, '\n')
    inds = np.where(theta!=0)[0]

    hessian1 = _ratematrix.hessian(theta, C, inds=inds)
    hessian2 = nd.Jacobian(lambda x: _ratematrix.loglikelihood(x, C)[1])(theta)
    hessian3 = nd.Hessian(lambda x: _ratematrix.loglikelihood(x, C)[0])(theta)

    np.set_printoptions(precision=3)

    #H1 = hessian1[np.ix_(active, active)]
    #H2 = hessian2[np.ix_(active, active)]
    #H3 = hessian2[np.ix_(active, active)]

    print(hessian1, '\n')
    print(hessian2, '\n')
    #print(hessian3)
    print('\n')


    info1 = np.zeros((len(theta), len(theta)))
    info2 = np.zeros((len(theta), len(theta)))
    info1[np.ix_(inds, inds)] = scipy.linalg.pinv(-hessian1)
    info2[np.ix_(inds, inds)] = scipy.linalg.pinv(-hessian2[np.ix_(inds, inds)])

    print('Inverse Hessian')
    print(info1)
    print(info2)
    #print(scipy.linalg.pinv(hessian2))
    #print(scipy.linalg.pinv(hessian1)[np.ix_(last, last)])
    #print(scipy.linalg.pinv(hessian2)[np.ix_(last, last)])

    print(_ratematrix.sigma_pi(info1, theta, n))
    print(_ratematrix.sigma_pi(info2, theta, n))

    #print(_ratematrix.sigma_pi(scipy.linalg.pinv(-hessian2), theta, n))
    #print(_ratematrix.sigma_pi(scipy.linalg.pinv(-hessian3), theta, n))

    # print(np.linalg.norm(H1-H2))
    #
    # # print(hessian1.shape)
    # # print(hessian1-hessian2)
    #
    # # not sure what the cutoff here should be (see plot_test_hessian)
    # assert np.linalg.norm(hessian1-hessian2) < 1e-6



def test_hessian_3():
    grid = NDGrid(n_bins_per_feature=4, min=-np.pi, max=np.pi)
    seqs = grid.fit_transform(load_doublewell(random_state=0)['trajectories'])
    seqs = [seqs[i] for i in range(10)]

    lag_time = 10
    model = ContinuousTimeMSM(verbose=True, lag_time=lag_time)
    model.fit(seqs)
    msm = MarkovStateModel(verbose=False, lag_time=lag_time)
    print(model.summarize())
    #print('MSM timescales\n', msm.fit(seqs).timescales_)
    print('Uncertainty K\n', model.uncertainty_K())
    print('Uncertainty eigs\n', model.uncertainty_eigenvalues())



def test_fit_1():
    # call fit, compare to MSM
    sequence = [0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2]
    model = ContinuousTimeMSM(verbose=False)
    model.fit([sequence])

    msm = MarkovStateModel(verbose=False)
    msm.fit([sequence])

    # they shouldn't be equal in general, but for this input they seem to be
    np.testing.assert_array_almost_equal(model.transmat_, msm.transmat_)


def test_fit_2():
    grid = NDGrid(n_bins_per_feature=5, min=-np.pi, max=np.pi)
    seqs = grid.fit_transform(load_doublewell(random_state=0)['trajectories'])

    model = ContinuousTimeMSM(verbose=True, lag_time=10)
    model.fit(seqs)
    t1 = np.sort(model.timescales_)
    t2 = -1/np.sort(np.log(np.linalg.eigvals(model.transmat_))[1:])

    model = MarkovStateModel(verbose=False, lag_time=10)
    model.fit(seqs)
    t3 = np.sort(model.timescales_)

    np.testing.assert_array_almost_equal(t1, t2)
    # timescales should be similar to MSM (withing 50%)
    assert abs(t1[-1] - t3[-1]) / t1[-1] < 0.50


def test_score_1():
    grid = NDGrid(n_bins_per_feature=5, min=-np.pi, max=np.pi)
    seqs = grid.fit_transform(load_doublewell(random_state=0)['trajectories'])
    model = ContinuousTimeMSM(verbose=False, lag_time=10, n_timescales=3).fit(seqs)
    np.testing.assert_approx_equal(model.score(seqs), model.score_)


def test_optimize_1():
    n = 100
    grid = NDGrid(n_bins_per_feature=n, min=-np.pi, max=np.pi)
    seqs = grid.fit_transform(load_doublewell(random_state=0)['trajectories'])

    model = ContinuousTimeMSM(verbose=True).fit(seqs)

    y, x, n = model.loglikelihoods_.T
    x = x-x[0]
    cross = np.min(np.where(n==n[-1])[0])


def test_score_2():
    from msmbuilder.example_datasets.muller import MULLER_PARAMETERS as PARAMS
    ds = MullerPotential(random_state=0).get()['trajectories']
    cluster = NDGrid(n_bins_per_feature=6,
          min=[PARAMS['MIN_X'], PARAMS['MIN_Y']],
          max=[PARAMS['MAX_X'], PARAMS['MAX_Y']])
    assignments = cluster.fit_transform(ds)
    test_indices = [5, 0, 4, 1, 2]
    train_indices = [3, 6, 7, 8, 9]

    model = ContinuousTimeMSM(lag_time=3, n_timescales=1)
    model.fit([assignments[i] for i in train_indices])
    test = model.score([assignments[i] for i in test_indices])
    train = model.score_
    print('train', train, 'test', test)
    assert 1 <= test < 2
    assert 1 <= train < 2


def test_score_3():
    import warnings
    warnings.simplefilter('ignore')
    from msmbuilder.example_datasets.muller import MULLER_PARAMETERS as PARAMS

    cluster = NDGrid(n_bins_per_feature=6,
          min=[PARAMS['MIN_X'], PARAMS['MIN_Y']],
          max=[PARAMS['MAX_X'], PARAMS['MAX_Y']])

    ds = MullerPotential(random_state=0).get()['trajectories']
    assignments = cluster.fit_transform(ds)

    train_indices = [9, 4, 3, 6, 2]
    test_indices = [8, 0, 5, 7, 1]

    model = ContinuousTimeMSM(lag_time=3, n_timescales=1, sliding_window=False, ergodic_cutoff=1)
    train_data = [assignments[i] for i in train_indices]
    test_data = [assignments[i] for i in test_indices]

    model.fit(train_data)
    train = model.score_
    test = model.score(test_data)
    print(train, test)
