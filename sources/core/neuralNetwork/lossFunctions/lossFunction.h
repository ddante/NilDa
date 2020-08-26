#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include "primitives/Scalar.h"
#include "primitives/Matrix.h"
#include "primitives/Vector.h"

#include "lossFunctionUtils.h"

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

  virtual void predict(
                       const Matrix& output,
                       Matrix& predictions
                      ) = 0;

  // Destructors

  virtual ~lossFunction() = default;

  virtual int type() const = 0;
};


} // namespace

#endif
