#ifndef MIXTAPE_DIAGONALGAUSSIANHMMESTEP_H
#define MIXTAPE_DIAGONALGAUSSIANHMMESTEP_H
#include "typedefs.hpp"
#include "DiagonalGaussianHMMSufficientStats.hpp"
#include "HMMEstep.hpp"

namespace Mixtape {

class DiagonalGaussianHMMEstep : public HMMEstep {
public:
    DiagonalGaussianHMMEstep(const int numStates,
                             const DoubleArray2D& transmat,
                             const DoubleArray1D& startProb,
                             const int numFeatures,
                             const DoubleArray2D& means,
                             const DoubleArray2D& variances);

    DiagonalGaussianHMMSufficientStats* initializeSufficientStats();
    FloatArray2D emissionLogLikelihood(const FloatArray2D& X);
    void accumulateSufficientStats(DiagonalGaussianHMMSufficientStats& stats, const FloatArray2D& seq, const FloatArray1D& frameLogProb,
                                   const FloatArray2D& posteriors, const DoubleArray2D& fwdLattice, const DoubleArray2D& bwdLattice);

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
