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
#ifdef ND_DEBUG_CHECKS
    assert(output.rows() == linearInput.rows());
    assert(output.cols() == linearInput.cols());    
#endif

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

    RowArray dotF = activation.cwiseProduct(G).colwise().sum();

    output.array() = activation.array() * (G.array().rowwise() - dotF);
}


} // namespace