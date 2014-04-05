#ifndef MIXTAPE_DIAGONALGAUSSIANHMMSUFFICIENTSTATS_H
#define MIXTAPE_DIAGONALGAUSSIANHMMSUFFICIENTSTATS_H
#include "typedefs.hpp"
#include "HMMSufficientStats.hpp"

namespace Mixtape {
class DiagonalGaussianHMMSufficientStats : public HMMSufficientStats {
public:
    DiagonalGaussianHMMSufficientStats(const int numStates)
        : HMMSufficientStats(numStates)
    { };


};

} // namespace
#endif
