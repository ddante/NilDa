#include "primitives/Array.h"
#include "softmax.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{


void softmax::applyForward(
                                    const Matrix& linearInput,
                                    Matrix& output
                                   )
{
    output.array() = linearInput.array().exp();

    RowArray sums = output.colwise().sum();

    output.array().rowwise() /= sums;
}

void softmax::applyBackward(
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

    RowArray dotF = activation.cwiseProduct(G).colwise().sum();

    output.array() = activation.array() * (G.array().rowwise() - dotF);
}


} // namespace