#include <iostream>
#include "primitives/Array.h"

#include "sparseCategoricalCrossentropy.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{


Scalar 
sparseCategoricalCrossentropy::compute(
                                                   const Matrix& obs, 
                                                   const Matrix& labels
                                                   )
{
    int nObs = obs.cols();

    RowArray s;
    s.resize(obs.rows());

    s = ((obs.array()).log() * labels.array()).colwise().sum()
      + ((1.0-obs.array()).log() * (1.0 - labels.array())).colwise().sum();
      
    return (-1.0/nObs)*s.sum();    
}

void
sparseCategoricalCrossentropy::computeDerivative(
                                                               const Matrix& obs, 
                                                               const Matrix& labels,
                                                               Matrix& output
                                                              )
{    
    output = -labels.array() * obs.array().cwiseInverse()
            + (1.0 - labels.array()) * (1.0 - obs.array()).cwiseInverse();
}


} // namespace