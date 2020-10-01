#ifndef SIGMOID_H
#define SIGMOID_H

#include "activationFunction.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


class sigmoid : public activationFunction
{

public:

  sigmoid() = default;

  void applyForward(
                    const Matrix& linearInput,
                    Matrix& output
                   ) override;

  void applyBackward(
                     const Matrix& linearInput,
                     const Matrix& G,
                     Matrix& output
                    ) override;

  int type() const override
  {
    return
      static_cast<std::underlying_type_t<activationFunctions> >(
        activationFunctions::sigmoid
      );
  }

  ~sigmoid() = default;
};

} // namespace

#endif
