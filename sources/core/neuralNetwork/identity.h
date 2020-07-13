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

    Matrix applyForward(const Matrix& linearInput) override;

    Matrix applyBackward(const Matrix& linearInput, const Matrix& G) override;

    ~identity() = default;
};


} // namespace

#endif