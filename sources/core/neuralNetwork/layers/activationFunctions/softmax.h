#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "activationFunction.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


class softmax : public activationFunction
{

public:

  softmax() = default;

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
        activationFunctions::softmax
      );
  }

  ~softmax() = default;
};


} // namespace

#endif
