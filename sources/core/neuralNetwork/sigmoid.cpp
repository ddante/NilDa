#include "sigmoid.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{


Matrix sigmoid::applyForward(const Matrix& linearInput)
{
    Matrix activation;

    activation.array() = 1.0/(1.0 + (-linearInput.array()).exp());

    return activation; 
}

Matrix sigmoid::applyBackward(const Matrix& linearInput, const Matrix& G)
{            
    Matrix jacobian;

    Matrix activation = applyForward(linearInput);

    jacobian.array() = activation.array()
                          * (1.0 - activation.array()) * G.array(); 

    return jacobian;
}


} // namespace