#ifndef SPARSE_CATEGORICAL_CROSSENTROPY_H
#define SPARSE_CATEGORICAL_CROSSENTROPY_H

#include "primitives/Scalar.h"
#include "primitives/Vector.h"
#include "primitives/Matrix.h"

#include "lossFunction.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{


class sparseCategoricalCrossentropy : public lossFunction
{
public:

    // Constructors

    sparseCategoricalCrossentropy() = default;

    // Member functions

    Scalar compute(
                        const Matrix& obs, 
                        const Matrix& labels
                       ) override;

    void computeDerivative(
                                 const Matrix& data, 
                                 const Matrix& labels, 
                                 Matrix& output
                                ) override; 
    
    // Destructor

    ~sparseCategoricalCrossentropy() = default;
};


} // namespace

#endif
