#include "relu.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{


void relu::applyForward(
                               const Matrix& linearInput,
                               Matrix& output
                              ) 
{
    output.array() = linearInput.array().max(0);
}

void relu::applyBackward(
                                const Matrix& linearInput, 
                                const Matrix& G,
                                Matrix& output
                               )
{          
    output.array() = (linearInput.array() > 0).select(G, 0);
}

} // namespace