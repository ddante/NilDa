#include "relu.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{


void relu::applyForward(
                               const Matrix& linearInput,
                               Matrix& output
                              ) 
{
#ifdef NILDA_DEBUG_BUILD    
    assert(output.rows() == linearInput.rows());
    assert(output.cols() == linearInput.cols());
#endif

    output.array() = linearInput.array().max(0);
}

void relu::applyBackward(
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
    
    output.array() = (linearInput.array() > 0).select(G, 0);
}

} // namespace