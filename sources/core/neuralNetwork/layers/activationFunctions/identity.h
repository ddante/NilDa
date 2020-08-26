#ifndef IDENTITY_H
#define IDENTITY_H

#include "activationFunction.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


class identity : public activationFunction
{
public:

  identity() = default;

  void applyForward(
                    const Matrix& linearInput,
                    Matrix& output
                   ) override;

  void applyBackward(
                     const Matrix& linearInput,
                     const Matrix& G,
                     Matrix& output
                    ) override;

  ~identity() = default;

  int type() const override
  {
    return
      static_cast<std::underlying_type_t<activationFucntions>>(
        activationFucntions::identity
      );
  }
};


} // namespace

#endif
