#ifndef MIXTAPE_TYPEDEFS_H
#define MIXTAPE_TYPEDEFS_H
#include <iostream>
#include <boost/multi_array.hpp>

namespace Mixtape {

typedef boost::multi_array<float, 2>  FloatArray2D;
typedef boost::multi_array<float, 1>  FloatArray1D;
typedef boost::multi_array<double, 2> DoubleArray2D;
typedef boost::multi_array<double, 1> DoubleArray1D;

} // namespace
#endif
