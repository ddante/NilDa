#ifndef ACTVIVATION_FUNCTION_H
#define ACTVIVATION_FUNCTION_H

#include <iostream>
#include "activationFunctionUtils.h"
#include "primitives/Vector.h"
#include "primitives/Matrix.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


class activationFunction
{
public:

  // Constructors

  activationFunction() = default;

  // Member functions

  // Apply the activation function in the forward propagation
  virtual void applyForward(
                            const Matrix& linearInput,
                            Matrix& output
                           ) = 0;

  // Apply the derivative of the activation function to G
  // in the backward propagation
  virtual void applyBackward(
                             const Matrix& linearInput,
                             const Matrix& G,
                             Matrix& output
                            ) = 0;

  // Destructor

  virtual ~activationFunction() = default;

  virtual int type() const = 0;
};

} // namespace

#endif
