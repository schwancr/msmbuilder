/*****************************************************************/
/*    Copyright (c) 2014, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#ifndef MIXTAPE_DIAGONALGAUSSIANHMMSUFFICIENTSTATS_H
#define MIXTAPE_DIAGONALGAUSSIANHMMSUFFICIENTSTATS_H
#include "typedefs.hpp"
#include "HMMEstep.hpp"
#include "HMMSufficientStats.hpp"

namespace Mixtape {
class DiagonalGaussianHMMSufficientStats : public HMMSufficientStats {
public:
    DiagonalGaussianHMMSufficientStats(const int numStates, const int numFeatures)
        : HMMSufficientStats(numStates)
        , numFeatures_(numFeatures)
        , posts_(boost::extents[numStates])
        , obs_(boost::extents[numStates][numFeatures])
        , obs2_(boost::extents[numStates][numFeatures])
    { };

    DoubleArray1D& posts() { return posts_; }
    DoubleArray2D& obs() { return obs_; }
    DoubleArray2D& obs2() { return obs2_; }

    void increment(const HMMEstep* estep,
                   const FloatArray2D& seq,
                   const FloatArray2D& frameLogProb,
                   const FloatArray2D& posteriors,
                   const DoubleArray2D& fwdLattice,
                   const DoubleArray2D& bwdLattice);

private:
    int numFeatures_;
    DoubleArray1D posts_;
    DoubleArray2D obs_;
    DoubleArray2D obs2_;
    

};

} // namespace
#endif
