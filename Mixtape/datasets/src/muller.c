#include "math.h"

#define SQR(x) ((x)*(x))

static const double MULLER_aa[4] = {-1, -1, -6.5, 0.7};
static const double MULLER_bb[4] = {0, 0, 11, 0.6};
static const double MULLER_cc[4] = {-10, -10, -6.5, 0.7};
static const double MULLER_AA[4] = {-200, -100, -170, 15};
static const double MULLER_XX[4] = {1, 0, -0.5, -1};
static const double MULLER_YY[4] = {0, 0.5, 1.5, 1};


double _muller_potential(double x[2], double beta)
{
    int j;
    double value = 0;

    for (j = 0; j < 4; j++) {
        value += MULLER_AA[j] * exp(
            MULLER_aa[j] * SQR(x[0] - MULLER_XX[j])
            + MULLER_bb[j] * (x[0] - MULLER_XX[j]) * (x[1] - MULLER_YY[j])
            + MULLER_cc[j] * SQR(x[1] - MULLER_YY[j]));
    }

    return beta * value;
}

void _muller_grad(const double x[2], double beta, double grad[2])
{
    int j;
    double value = 0;
    double term;
    grad[0] = 0;
    grad[1] = 0;

    for (j = 0; j < 4; j++) {
        term = MULLER_AA[j] * exp(
            MULLER_aa[j] * SQR(x[0] - MULLER_XX[j])
            + MULLER_bb[j] * (x[0] - MULLER_XX[j]) * (x[1] - MULLER_YY[j])
            + MULLER_cc[j] * SQR(x[1] - MULLER_YY[j]));

        grad[0] += (2 * MULLER_aa[j] * (x[0] - MULLER_XX[j])
                + MULLER_bb[j] * (x[1] - MULLER_YY[j])) * term;
        grad[1] += (MULLER_bb[j] * (x[0] - MULLER_XX[j])
                + 2 * MULLER_cc[j] * (x[1] - MULLER_YY[j])) * term;
    }
    
    grad[0] *= beta;
    grad[1] *= beta;
}