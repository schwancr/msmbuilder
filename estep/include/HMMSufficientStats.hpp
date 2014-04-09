#ifndef MIXTAPE_HMMSUFFICIENTSTATS_H
#define MIXTAPE_HMMSUFFICIENTSTATS_H
#include "typedefs.hpp"
#include "HMMEstep.hpp"

namespace Mixtape {
class HMMSufficientStats {
public:
    HMMSufficientStats(int numStates)
        : numStates_(numStates)
        , logProb_(0)
        , startCounts_(boost::extents[numStates])
        , transCounts_(boost::extents[numStates][numStates])
    { };

    int numObservations() { return numObservations_; }
    void numObservations(int numObservations) { numObservations_ = numObservations; }
    double logProb() { return logProb_; }
    DoubleArray1D& startCounts() { return startCounts_; }
    DoubleArray2D& transCounts() { return transCounts_; }

    virtual void increment(const HMMEstep* estep,
                           const FloatArray2D& seq,
                           const FloatArray2D& frameLogProb,
                           const FloatArray2D& posteriors,
                           const DoubleArray2D& fwdLattice,
                           const DoubleArray2D& bwdLattice);

protected:
    int numStates_;

private:
    int numObservations_;
    double logProb_;
    DoubleArray1D startCounts_;
    DoubleArray2D transCounts_;
};


} // namespace
#endif
