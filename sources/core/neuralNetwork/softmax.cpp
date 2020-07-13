#include "primitives/Array.h"
#include "softmax.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{


Matrix softmax::applyForward(const Matrix& linearInput)
{
    Matrix activation;

    activation.array() = linearInput.array().exp();

    RowArray sums = activation.colwise().sum();

    activation.array().rowwise() /= sums;

    return activation;
}

Matrix softmax::applyBackward(const Matrix& linearInput, const Matrix& G)
{            
    Matrix jacobian;

    Matrix activation = applyForward(linearInput);

    RowArray dotF = activation.cwiseProduct(G).colwise().sum();

    jacobian.array() = activation.array() * (G.array().rowwise() - dotF);

    return jacobian;
}


} // namespace