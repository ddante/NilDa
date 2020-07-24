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

    virtual void get()  const = 0;

    // Initialize the map for the history of the gradients
    virtual void init(
                        const Matrix& weightsGradient, 
                        const Vector& biasesGradient
                       ) const = 0;

    // Compute the increments of the weights and biases
    virtual void update(
                            const Matrix& weightsGradient, 
                            const Vector& biasesGradient,
                            Matrix& deltaWeights,
                            Vector& deltaBiases
                           ) const = 0;

    // Destructor

    virtual ~optimizer() = default;
    
};


} // namespace

#endif