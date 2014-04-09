#ifndef MIXTAPE_LOGSUMEXP_H
#define MIXTAPE_LOGSUMEXP_H

double logsumexp(const double* __restrict__ buf, int N);
float logsumexp(const float* __restrict__ buf, int N);

#endif
