#include <iostream>

#include "primitives/Array.h"

#include "sgd.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{

sgd::sgd(Scalar alpha):
    learningRate_(alpha),
    momentum_(0.0)
{
    if (learningRate_ <= 0)
    {
        std::cerr << "Learning rate must be > 0.\n";
        assert(false);        
    }
}

sgd::sgd(Scalar alpha, Scalar m):
    learningRate_(alpha),
    momentum_(m)
{
    if (learningRate_ <= 0)
    {
        std::cerr << "In SGD optimizer, "
                     << "learning rate must be > 0.\n";
        assert(false);        
    }

    if (momentum_ < 0 || momentum_ >=1)
    {
        std::cerr << "In SGD optimizer, "
                          "momentum must be >= 0 and < 1 " << std::endl;
        assert(false);        
    }
}

void sgd::init(
                 const Matrix& weightsGradient, 
                 const Vector& biasesGradient
                ) const
{    
    Matrix& weightsFirstMomentum 
        = weightsHistory_[weightsGradient.data()];

    weightsFirstMomentum.resize(
                                        weightsGradient.rows(), 
                                        weightsGradient.cols()
                                       );

    weightsFirstMomentum.setZero();

    Vector& biasesFirstMomentum 
        = biasesHistory_[biasesGradient.data()];

    biasesFirstMomentum.resize(biasesGradient.rows());
    biasesFirstMomentum.setZero();    
}

void sgd::update(const Matrix& weightsGradient,
                     const Vector& biasesGradient,              
                     Matrix& deltaWeights,
                     Vector& deltaBiases
                    ) const
{   
    // Get the history of the weights and biases associated with  the current layer
    Matrix& weightsFirstMomentum 
        = weightsHistory_[weightsGradient.data()];

    Vector& biasesFirstMomentum 
        = biasesHistory_[biasesGradient.data()];

    // Update the weights and biases using momentum
    weightsFirstMomentum = momentum_ * weightsFirstMomentum
                                + (1.0 - momentum_) * weightsGradient;

    biasesFirstMomentum = momentum_ * biasesFirstMomentum
                              + (1.0 - momentum_) * biasesGradient;

    // Return the increment of the weights and biases
    deltaWeights.noalias() = -learningRate_ * weightsFirstMomentum;

    deltaBiases.noalias() = -learningRate_ * biasesFirstMomentum;
}


} // namespace