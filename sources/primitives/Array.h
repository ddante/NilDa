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


} // namespace

#endif
