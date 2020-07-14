#include "identity.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{


Matrix identity::applyForward(const Matrix& linearInput)
{
    return linearInput; 
}

Matrix identity::applyBackward(const Matrix& linearInput, const Matrix& G)
{          
    return G;
}

} // namespace