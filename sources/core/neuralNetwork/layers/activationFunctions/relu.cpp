#include "relu.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{


Matrix relu::applyForward(const Matrix& linearInput)
{
    Matrix activation;

    activation.array() = linearInput.array().max(0);

    return activation;
}

Matrix relu::applyBackward(const Matrix& linearInput, const Matrix& G)
{            
    Matrix jacobian;

    jacobian.array() = (linearInput.array() > 0).select(G, 0);

    return jacobian;
}

} // namespace