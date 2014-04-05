#ifndef MIXTAPE_HMMESTEP_H
#define MIXTAPE_HMMESTEP_H
#include <map>
#include <vector>
#include "typedefs.hpp"
#include "HMMSufficientStats.hpp"
#include "MixtapeException.hpp"
namespace Mixtape {

class HMMEstep {
public:
    /*
    Construct a HMMEstep
    */
    HMMEstep(const int numStates, const DoubleArray2D& transMat, const DoubleArray1D& startProb);

    /*
    Register a new data sequence with the model
    */
    void addSequence(const FloatArray2D* X);


    DoubleArray2D forwardPass(const FloatArray2D& frameLogProb);
    DoubleArray2D backwardPass(const FloatArray2D& frameLogProb);
    FloatArray2D computePosteriors(const DoubleArray2D& fwdLattice, const DoubleArray2D& bwdLattice);
    HMMSufficientStats execute();


    virtual FloatArray2D emissionLogLikelihood(const FloatArray2D& X) = 0;
    virtual HMMSufficientStats* initializeSufficientStats();
    virtual void accumulateSufficientStats(HMMSufficientStats& stats,
        const FloatArray2D& seq, const FloatArray2D& frameLogProb,
        const FloatArray2D& posteriors, const DoubleArray2D& fwdLattice,
        const DoubleArray2D& bwdLattice);

    void print(const DoubleArray2D& t) {
        for (int i = 0; i < t.shape()[0]; i++) {
            for (int j = 0; j < t.shape()[1]; j++) {
                std::cout << t[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
    void print(const FloatArray2D& t) {
        for (int i = 0; i < t.shape()[0]; i++) {
            for (int j = 0; j < t.shape()[1]; j++) {
                std::cout << t[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }




protected:
    int numStates_;

private:
    std::vector<const FloatArray2D*> sequences_;
    DoubleArray2D logTransmat_;
    DoubleArray2D logTransmat_T_;
    DoubleArray1D logStartProb_;
};



} // namespace

#endif
