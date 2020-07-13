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

    Matrix applyForward(const Matrix& linearInput) override;

    Matrix applyBackward(const Matrix& linearInput, const Matrix& G) override;

    ~relu()  = default;
};

} // namespace

#endif