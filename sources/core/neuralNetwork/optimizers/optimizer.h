#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "primitives/Vector.h"
#include "primitives/Matrix.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{

class optimizer
{
public:

    // Constructors

    optimizer() = default;

    // Member functions 

    // Initialize the map for the history of the gradients
    virtual void init(
                        const Matrix& weightsGradient, 
                        const Vector& biasesGradient
                       ) = 0;

    // Compute the increments of the weights and biases
    virtual void update(
                            const Matrix& weightsGradient, 
                            const Vector& biasesGradient,
                            Matrix& deltaWeights,
                            Vector& deltaBiases
                           ) = 0;

    // Destructor

    virtual ~optimizer() = default;
    
};


} // namespace

#endif