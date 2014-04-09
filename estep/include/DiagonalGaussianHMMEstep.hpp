#ifndef MIXTAPE_DIAGONALGAUSSIANHMMESTEP_H
#define MIXTAPE_DIAGONALGAUSSIANHMMESTEP_H
#include "typedefs.hpp"
#include "HMMEstep.hpp"
#include "DiagonalGaussianHMMSufficientStats.hpp"

namespace Mixtape {

class DiagonalGaussianHMMEstep : public HMMEstep {
public:
    DiagonalGaussianHMMEstep(const int numStates,
                             const DoubleArray2D& transmat,
                             const DoubleArray1D& startProb,
                             const int numFeatures,
                             const DoubleArray2D& means,
                             const DoubleArray2D& variances);

    virtual DiagonalGaussianHMMSufficientStats* initializeSufficientStats();
    virtual FloatArray2D emissionLogLikelihood(const FloatArray2D& X);

protected:
    const int numFeatures_;

private:
    const DoubleArray2D means_;
    const DoubleArray2D variances_;
    DoubleArray2D logVariances_;
    DoubleArray2D meansOverVariances_;
    DoubleArray2D means2OverVariances_;    
};


} // namespace
#endif
