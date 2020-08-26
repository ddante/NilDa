#ifndef RELU_H
#define RELU_H

#include "activationFunction.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


class relu : public activationFunction
{
public:

  relu() = default;

  void applyForward(
                    const Matrix& linearInput,
                    Matrix& output
                   ) override;

  void applyBackward(
                     const Matrix& linearInput,
                     const Matrix& G,
                     Matrix& output
                    ) override;

  ~relu()  = default;

  int type() const override
  {
    return
      static_cast<std::underlying_type_t<activationFucntions>>(
        activationFucntions::relu
      );
  }
};

} // namespace

#endif
