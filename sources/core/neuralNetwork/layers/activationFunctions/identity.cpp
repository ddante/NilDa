#include "identity.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{


void identity::applyForward(
                                   const Matrix& linearInput,
                                   Matrix& output
                                  )
{
#ifdef NILDA_DEBUG_BUILD    
    assert(output.rows() == linearInput.rows());
    assert(output.cols() == linearInput.cols());
#endif

    output.noalias() = linearInput; 
}

void identity::applyBackward(
                                     const Matrix& linearInput, 
                                     const Matrix& G,
                                     Matrix& output
                                    )
{          
#ifdef NILDA_DEBUG_BUILD
    assert(G.rows() == linearInput.rows());
    assert(G.cols() == linearInput.cols());    
    
    assert(output.rows() == linearInput.rows());
    assert(output.cols() == linearInput.cols());
#endif
    
    output.noalias() = G;
}

} // namespace