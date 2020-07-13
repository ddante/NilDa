#ifndef ACTVIVATION_FUNCTION_H
#define ACTVIVATION_FUNCTION_H

# include <iostream>

#include "primitives/Vector.h"
#include "primitives/Matrix.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{


class activationFunction
{
public:

    // Constructors

    activationFunction() = default;

    // Member functions

    // Apply the activation function in the forward propagation
    virtual Matrix applyForward(const Matrix& linearInput) = 0;

    // Apply the derivative of the activation function to G
    // in the backward propagation
    virtual Matrix applyBackward(const Matrix& linearInput, const Matrix& G) = 0;

    // Destructor

    virtual ~activationFunction() {};
    
};
 
} // namespace

#endif