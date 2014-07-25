import numpy as np
from sklearn.utils import check_random_state
from libc.math cimport exp, sqrt
cimport cython


cdef double *MULLER_aa = [-1, -1, -6.5, 0.7]
cdef double *MULLER_bb = [0, 0, 11, 0.6]
cdef double *MULLER_cc = [-10, -10, -6.5, 0.7]
cdef double *MULLER_AA = [-200, -100, -170, 15]
cdef double *MULLER_XX = [1, 0, -0.5, -1]
cdef double *MULLER_YY = [0, 0.5, 1.5, 1]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double _muller_potential(double[::1] x, double beta=1):
    """Muller potential. This function is relative low-level and
    takes a input a single point as a length-2 double array.
    """
    cdef int j
    cdef double value = 0
    assert len(x) == 2

    for j in range(4):
        value += MULLER_AA[j] * exp(
            MULLER_aa[j] * (x[0] - MULLER_XX[j])**2
            + MULLER_bb[j] * (x[0] - MULLER_XX[j]) * (x[1] - MULLER_YY[j])
            + MULLER_cc[j] * (x[1] - MULLER_YY[j])**2)

    return beta * value

def muller_potential(x, y):
    """Muller potential.
    
    This function can take vectors (x,y) as input and will broadcast
    """
    return np.vectorize(lambda x, y: _muller_potential(np.array([x,y])))(x, y)


cdef _muller_grad(double[::1] x, double beta, double[::1] grad):
    """Low-level code for grad of the muller potential. This takes a
    single length-2 double array as input in `x`, and writes the grad
    to `grad`. 
    """
    cdef int j
    cdef double value = 0
    cdef double term
    assert len(x) == 2
    assert len(grad) == 2
    grad[0] = 0
    grad[1] = 0

    for j in range(4):
        term = MULLER_AA[j] * exp(
            MULLER_aa[j] * (x[0] - MULLER_XX[j])**2
            + MULLER_bb[j] * (x[0] - MULLER_XX[j]) * (x[1] - MULLER_YY[j])
            + MULLER_cc[j] * (x[1] - MULLER_YY[j])**2)

        grad[0] += (2 * MULLER_aa[j] * (x[0] - MULLER_XX[j])
                + MULLER_bb[j] * (x[1] - MULLER_YY[j])) * term
        grad[1] += (MULLER_bb[j] * (x[0] - MULLER_XX[j])
                + 2 * MULLER_cc[j] * (x[1] - MULLER_YY[j])) * term

    grad[0] *= beta
    grad[1] *= beta


@cython.boundscheck(False)
@cython.wraparound(False)
def propagate(n_steps=5000, x0=[-0.5, 0.5], thin=1,
              double kT=1.5e4, double dt=0.1, double D=0.010,
              random_state=None):
    cdef int i, j
    cdef double DT_SQRT_2D = dt * sqrt(2 * D)
    cdef double beta = 1.0 / kT
    random = check_random_state(random_state)
    cdef double[:, ::1] r = random.randn(n_steps, 2)
    cdef double[:, ::1] saved_x = np.zeros(((n_steps)/thin, 2))
    cdef double x[2]
    cdef double grad[2]

    x[0] = x0[0]
    x[1] = x0[1]
    
    cdef int save_index = 0
    for i in range(n_steps):
        _muller_grad(x, beta, grad)
        for j in range(2):
            x[j] = x[j] - dt * grad[j] + DT_SQRT_2D * r[i, j]
        if i % thin == 0:
            saved_x[save_index] = x
            save_index += 1

    return np.asarray(saved_x)

