#ifndef ARRAY_H
#define ARRAY_H

#include <Eigen/Core>
#include <Eigen/Dense>

#include "Scalar.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


typedef Eigen::Array<
                     Scalar,
                     1,
                     Eigen::Dynamic
                    >
                    RowArray;

typedef Eigen::Array<
                     Scalar,
                     Eigen::Dynamic,
                     1
                    >
                    ColArray;

typedef Eigen::Array<
                     int,
                     1,
                     Eigen::Dynamic
                    >
                    RowArrayI;

typedef Eigen::Array<
                     int,
                     Eigen::Dynamic,
                     1
                    >
                    ColArrayI;


} // namespace

#endif
