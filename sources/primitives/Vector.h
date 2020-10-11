#ifndef VECTOR_H
#define VECTOR_H

#include <Eigen/Core>
#include <Eigen/Dense>

#include "Scalar.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

typedef Eigen::Matrix<int, Eigen::Dynamic, 1> VectorI;

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

typedef Eigen::Map<const Vector> ConstMapVector;

typedef Eigen::Map<Vector> MapVector;

} // namespace

#endif
