#ifndef CATEGORICAL_CROSSENTROPY_H
#define CATEGORICAL_CROSSENTROPY_H

#include "primitives/Scalar.h"
#include "primitives/Vector.h"
#include "primitives/Matrix.h"

#include "lossFunction.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


class categoricalCrossentropy : public lossFunction
{
public:

  // Constructors

  categoricalCrossentropy() = default;

  // Member functions

  Scalar compute(
                 const Matrix& obs,
                 const Matrix& labels
                ) override;

  void computeDerivative(
                         const Matrix& data,
                         const Matrix& labels,
                         Matrix& output
                        ) override;

  void predict(
               const Matrix& output,
               Matrix& predictions
              ) override;

  // Destructor

  ~categoricalCrossentropy() = default;

  int type() const override
  {
    return
      static_cast<std::underlying_type_t<lossFunctions>>(
        lossFunctions::categoricalCrossentropy
      );
  }
};


} // namespace

#endif
