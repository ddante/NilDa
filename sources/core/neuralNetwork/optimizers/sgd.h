#ifndef SGD_H
#define SGD_H

#include <map>

#include "optimizer.h"

#include "primitives/Scalar.h"
#include "primitives/Array.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{

class sgd : public optimizer
{

private:

    Scalar learningRate_;

    Scalar momentum_;

    // To recover the correct history of the weights and
    // biases associated with their respective gradient,
    // store the history in map using as key a constant pointer
    // to the gradients of the weights and biases
    std::map<const Scalar*, Matrix> weightsHistory_;

    std::map<const Scalar*, Vector> biasesHistory_;

public:

    // Constructors

    sgd(Scalar alpha);

    sgd(Scalar alpha, Scalar m);

    void init(
               const Matrix& weightsGradient, 
               const Vector& biasesGradient
              ) override;

    // Member functions
    void update(const Matrix& weightsGradient, 
                   const Vector& biasesGradient,                   
                   Matrix& deltaWeights,
                   Vector& deltaBiases
                  ) override;

    // Destructor
    ~sgd() = default;

    
};


} // namespace

#endif