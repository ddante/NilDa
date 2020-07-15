#include "identity.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{


void identity::applyForward(
                                   const Matrix& linearInput,
                                   Matrix& output
                                  )
{
    output.noalias() = linearInput; 
}

void identity::applyBackward(
                                     const Matrix& linearInput, 
                                     const Matrix& G,
                                     Matrix& output
                                    )
{          
    output.noalias() = G;
}

} // namespace