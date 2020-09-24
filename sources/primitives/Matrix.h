#ifndef MATRIX_H
#define MATRIX_H

#include <Eigen/Core>
#include <Eigen/Dense>

#include <random>

#include "Scalar.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


typedef Eigen::Matrix<
                      int,
                      Eigen::Dynamic,
                      Eigen::Dynamic
                     >
                     MatrixI;

typedef Eigen::Matrix<
                      Scalar,
                      Eigen::Dynamic,
                      Eigen::Dynamic
                     >
                     Matrix;

typedef Eigen::Matrix<
                      Scalar,
                      Eigen::Dynamic,
                      Eigen::Dynamic,
                      Eigen::RowMajor
                     >
                     RowMatrix;

typedef Eigen::Map<const Matrix> ConstMapMatrix;

typedef Eigen::Map<Matrix> MapMatrix;

} // namespace

#endif
