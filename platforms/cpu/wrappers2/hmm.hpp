#ifndef MIXTAPE_CPU_GHMM_ESTEP
#define MIXTAPE_CPU_GHMM_ESTEP

#include "MixtapeException.hpp"
#include "forward.hpp"
#include "backward.hpp"
#include <boost/multi_array.hpp>
#include <iostream>
#include <numeric>
#include <vector>
#include <map>


#ifdef _OPENMP
#include "omp.h"
#endif


namespace Mixtape {

class HiddenMarkovModelEstep {
public:
    typedef boost::multi_array<float, 2>  FloatArray2D;
    typedef boost::multi_array<double, 2> DoubleArray2D;
    typedef boost::multi_array<double, 1> DoubleArray1D;

    /* 
    Construct a HiddenMarkovModelEstep
    */ 
    HiddenMarkovModelEstep(const int n_states_)
        : n_states(n_states_) {};

    /*
    Register a new data sequence with the model
    */
    void addSequence(const FloatArray2D& X) {
        sequences.push_back(X);
    }

    /*
    Set the transition matrix
    */
    void setTransmat(const DoubleArray2D& transmat_) {
        if (n_states != transmat.shape()[0] || n_states != transmat.shape()[1])
            throw MixtapeException("transmat has wrong shape");

        transmat = transmat;
        transmat_T(boost::extents[n_states][n_states]);
        
        for (int i = 0; i < n_states; i++)
            for (int j = 0; j < n_states; j++)
                transmat_T[j][i] = transmat[i][j];
    }
    
    /*
    Get the transition matrix
    */
    DoubleArray2D getTransmat() {
        return transmat;
    }
    
    /*
    Set the startprob
    */
    void setStartProb(const DoubleArray1D& startProb_) {
        if (n_states != startProb_.shape()[0])
            throw MixtapeException("startprob has wrong shape");
        startProb = startProb_;
    }
    
    /*
    Get the startprob
    */
    DoubleArray1D getStartProb() {
        return startProb;
    }
    
    
    /*
    Abstract method to compute the emission log likelohood for each data
    point in a sequence. Should be overridden in the subclass.
    */ 
    virtual DoubleArray1D emissionLogLikelihood(const FloatArray2D& X) = 0;
    // void accumulateSufficientStatistics(std::map<std::string, void*>& stats,
    //    const FloatArray2D& seq, const DoubleArray1D& frameLogProb,
    //    const DoubleArray2D& posteriors, const DoubleArray2D& fwdLattice,
    //    const DoubleArray2D& bwdLattice) {
    //    }

    std::map<std::string, void*> execute();
    
    

private:
    int n_states;
    std::vector<FloatArray2D> sequences;
    DoubleArray2D transmat;
    DoubleArray2D transmat_T;
    DoubleArray1D startProb;
};



} // namespace

#endif