#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include "primitives/Scalar.h"
#include "primitives/Matrix.h"
#include "primitives/Vector.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


class lossFunction
{
public:

  // Constructors

  lossFunction() = default;

  // Member functions

  virtual Scalar compute(
                         const Matrix& obs,
                         const Matrix& labels
                        ) = 0;

  virtual void computeDerivative(
                                 const Matrix& obs,
                                 const Matrix& labels,
                                 Matrix& output
                                ) = 0;

  // Destructors

  virtual ~lossFunction() = default;
};


} // namespace

#endif
