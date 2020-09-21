#ifndef BINARY_CROSSENTROPY_H
#define BINARY_CROSSENTROPY_H

#include "primitives/Scalar.h"
#include "primitives/Vector.h"
#include "primitives/Matrix.h"

#include "lossFunction.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


class binaryCrossentropy : public lossFunction
{
public:

  // Constructors

  binaryCrossentropy() = default;

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

  void sanityCheck(
                   const int outputSize,
                   const Matrix& labels
                  ) const override;

  // Destructor

  ~binaryCrossentropy() = default;

  int type() const override
  {
    return
      static_cast<std::underlying_type_t<lossFunctions>>(
        lossFunctions::binaryCrossentropy
      );
  }
};


} // namespace

#endif
