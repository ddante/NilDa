#include "sigmoid.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{


void sigmoid::applyForward(
                                   const Matrix& linearInput,
                                   Matrix& output
                                  )
{
#ifdef ND_DEBUG_CHECKS
    assert(output.rows() == linearInput.rows());
    assert(output.cols() == linearInput.cols());
#endif

    output.array() = 1.0/(1.0 + (-linearInput.array()).exp());
}

void sigmoid::applyBackward(
                                     const Matrix& linearInput, 
                                     const Matrix& G,
                                     Matrix& output
                                    )
{   
#ifdef ND_DEBUG_CHECKS
    assert(G.rows() == linearInput.rows());
    assert(G.cols() == linearInput.cols());    
    
    assert(output.rows() == linearInput.rows());
    assert(output.cols() == linearInput.cols());
#endif
         
    Matrix activation;
    activation.resize(
                          linearInput.rows(), 
                          linearInput.cols()
                         );
    
    applyForward(linearInput, activation);

    output.array() = activation.array()
                       * (1.0 - activation.array()) * G.array(); 
}


} // namespace