#include "sigmoid.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{


void sigmoid::applyForward(
                                   const Matrix& linearInput,
                                   Matrix& output
                                  )
{
    output.array() = 1.0/(1.0 + (-linearInput.array()).exp());
}

void sigmoid::applyBackward(
                                     const Matrix& linearInput, 
                                     const Matrix& G,
                                     Matrix& output
                                    )
{            
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