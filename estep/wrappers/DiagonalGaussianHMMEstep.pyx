import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "../include/DiagonalGaussianHMMEstep.hpp":
    cdef doDiagonalGaussianHMMEstep "Mixtape::doDiagonalGaussianHMMEstep"(
        const int numStates,
        const double* transmat,
        const double* startProb,
        const int numFeatures,
        const double* means,
        const double* variances,
        const float** sequences,
        float* transmat,
        float* posts,
        float* obs,
        float* obs2)
                                


def doDiagonalGaussianHMMEstep(
    int numStates,
    np.ndarray[ndim=2, mode='c', dtype=np.float64_t] transmat not None,
    np.ndarray[ndim=1, dtype=np.float64_t] startProb,
    int numFeatures,
    np.ndarray[ndim=2, dtype=np.float64_t] means,
    np.ndarray[ndim=2, dtype=np.float64_t] variances):
   
    pass
        
        
