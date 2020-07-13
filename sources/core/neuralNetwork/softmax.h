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

    Matrix applyForward(const Matrix& linearInput) override;

    Matrix applyBackward(const Matrix& linearInput, const Matrix& G) override;

    ~softmax() = default;
};


} // namespace

#endif