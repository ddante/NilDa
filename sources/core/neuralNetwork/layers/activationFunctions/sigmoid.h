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

    Matrix applyForward(const Matrix& linearInput) override;

    Matrix applyBackward(const Matrix& linearInput, const Matrix& G) override;

    ~sigmoid() = default;
};

} // namespace

#endif