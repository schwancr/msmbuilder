#ifndef MIXTAPE_HMMSUFFICIENTSTATS_H
#define MIXTAPE_HMMSUFFICIENTSTATS_H
#include "typedefs.hpp"

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

    void incrementLogProb(double logProb) {
        logProb_ += logProb;
    }
    void incrementTransCounts(const DoubleArray2D& transCounts) {
        typedef DoubleArray2D::index index;
        for (index i = 0; i < numStates_; i++)
            for (index j = 0; j < numStates_; j++)
                transCounts_[i][j] += transCounts[i][j];
    }

 

private:
    int numStates_;
    int numObservations_;
    double logProb_;
    DoubleArray1D startCounts_;
    DoubleArray2D transCounts_;
};


} // namespace
#endif
