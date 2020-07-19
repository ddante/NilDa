#include <iostream>
#include <primitives/errors.h>

#include "neuralNetwork.h"

#include "lossFunctions/sparseCategoricalCrossentropy.h"
#include "lossFunctions/lossFunctionUtils.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{


neuralNetwork::neuralNetwork(const std::vector<layer*>& vLayers):
    layers_(vLayers),
    numberOfLayers_(vLayers.size()),
    lastLayer_(numberOfLayers_ - 1),
    validState_(false)
{
     // The first layer must be an input layer
     if (layers_[0]->layerType() != layerTypes::input)
     {
         std::cerr << "First layer is not an input layer." << std::endl;
         assert(false);
     }

    // Initialize the hidden and the output layers
    for (int i = 1; i < numberOfLayers_; ++i)
    {
        layers_[i]->init(layers_[i - 1]);
    }
}

void neuralNetwork::forwardPropagation(const Matrix& obs) const
{   
    validState_ = false;

    // The layer 0 is an input layer, just check that
    // the size of the input data is consistent with the 
    // input layer size 
    layers_[0]->checkInputSize(obs);

    // The first actual layer takes in directly the input data
    layers_[firstLayer_]->forwardPropagation(obs);

    for (int i = 2; i < numberOfLayers_; ++i)
    {
        // The other layers take in the output of the previous layer
        layers_[i]->forwardPropagation(layers_[i - 1]->output());
    }

    validState_ = true;
}

void neuralNetwork::backwardPropagation(
                                                    const Matrix& obs, 
                                                    const Matrix& labels
                                                    ) const
{   
#ifdef ND_DEBUG_CHECKS
    assert(validState_);
#endif

    // Derivative of the cost function w.r.t the activation 
    // output of the last layer
    Matrix dLoss;
    dLoss.resize(labels.rows(), labels.cols());

    lossFunction_->computeDerivative(
                                                layers_[lastLayer_]->output(),
                                                labels, 
                                                dLoss
                                               );
    
    // Derivatives of the cost function w.r.t. the 
    // weights and biases of the last layer
    layers_[lastLayer_]->backwardPropagation(
                                                          dLoss, 
                                                          layers_[lastLayer_ - 1]->output()
                                                         );

    // Maybe the if statement is not necessary
    if(numberOfLayers_ > 2) 
    {        
        // Derivatives of the cost function w.r.t. the 
        // weights and biases for the hidden layers
        for (int i = lastLayer_ - 1; i > firstLayer_; --i)
        {    
            layers_[i]->backwardPropagation(
                                                      layers_[i + 1]->backPropCache(),
                                                      layers_[i - 1]->output()
                                                     );     
        }
    }

    // Derivatives of the cost function w.r.t. the 
    // weights and biases of the first layer
    layers_[firstLayer_]->backwardPropagation(
                                                           layers_[firstLayer_ + 1]->backPropCache(),
                                                           obs
                                                          );    
}

void neuralNetwork::setLossFunction(const std::string& lossName)
{
    switch(lossFunctionCode(lossName))
    {
        case lossFunctions::sparseCategoricalCrossentropy :
            lossFunction_ = std::make_unique<sparseCategoricalCrossentropy>();
            break;
        default :
            std::cerr << "Not valid loss function  " 
                        << lossName
                        << " in this context." << std::endl;
        assert(false);
    }
}

Scalar neuralNetwork::getLoss(
                                       const Matrix& obs, 
                                       const Matrix& labels
                                      ) const 
{   
    if(!validState_)
    {   
        std::cout << "Warning: need to apply forwardPropagation" << std::endl;
        forwardPropagation(obs);
    }  

    return lossFunction_->compute(
                                            layers_[lastLayer_]->output(), 
                                            labels
                                           );
}

errorCheck 
neuralNetwork::checkWeightsGradients(
                                                const int layer,
                                                const Matrix& obs, 
                                                const Matrix& labels,
                                                const Scalar eps,
                                                const Scalar errorLimit
                                               ) const
{       
    // Store the original weights and biases 
    const Matrix weightsBk = layers_[layer]->getWeights();

    const Vector biasesBk = layers_[layer]->getBiases(); 

    Matrix weights(weightsBk.rows(), weightsBk.cols());

    Matrix dWeightsNum(weights.rows(), weights.cols());

    for (int i = 0; i < weights.rows(); ++i)
    {
        for (int j = 0; j < weights.cols(); ++j)
        {
            // Compute the W+eps part
            weights.noalias() = weightsBk;
            weights(i, j) += eps;

            layers_[layer]->setWeightsAndBiases(
                                                           weights, 
                                                           biasesBk
                                                          );

            forwardPropagation(obs);

            const Scalar Jp = getLoss(obs, labels);

            // Compute the W-eps part
            weights.noalias() = weightsBk;
            weights(i, j) -= eps;

            layers_[layer]->setWeightsAndBiases(
                                                           weights, 
                                                           biasesBk
                                                          );
            
            forwardPropagation(obs);

            const Scalar Jm = getLoss(obs, labels);

            // Numerical gradients: central difference
            dWeightsNum(i,j) = (Jp - Jm)/(2*eps);
        }
    }

    // Restore the correct values of weights and biases
    layers_[layer]->setWeightsAndBiases(
                                                   weightsBk, 
                                                   biasesBk
                                                  );

    // Analytical derivative of the weights
    const Matrix dWeights = layers_[layer]->getWeightsDerivative();

    const Scalar error = (dWeights - dWeightsNum).norm() 
                           / (dWeights.norm() + dWeightsNum.norm());

    errorCheck output;
    output.code = (error < errorLimit) ? EXIT_OK : EXIT_FAIL;
    output.error = error;

    return output;
}

errorCheck 
neuralNetwork::checkBiasesGradients(
                                              const int layer,
                                              const Matrix& obs, 
                                              const Matrix& labels,
                                              const Scalar eps,
                                              const Scalar errorLimit
                                             ) const
{       
    // Store the original weights and biases 
    const Matrix weightsBk = layers_[layer]->getWeights();

    const Vector biasesBk = layers_[layer]->getBiases(); 

    Vector biases(biasesBk.rows());

    Vector dBiasesNum(biases.rows());

    for (int i = 0; i < biases.rows(); ++i)
    {
        // Compute the W+eps part
        biases.noalias() = biasesBk;
        biases(i) += eps;

        layers_[layer]->setWeightsAndBiases(
                                                       weightsBk, 
                                                       biases
                                                      );

        forwardPropagation(obs);

        const Scalar Jp = getLoss(obs, labels);

        // Compute the W-eps part
        biases.noalias() = biasesBk;
        biases(i) -= eps;

        layers_[layer]->setWeightsAndBiases(
                                                       weightsBk, 
                                                       biases
                                                      );

        forwardPropagation(obs);

        const Scalar Jm = getLoss(obs, labels);

        // Numerical gradients: central difference
        dBiasesNum(i) = (Jp - Jm)/(2*eps);
    }

    // Restore the correct values of weights and biases
    layers_[layer]->setWeightsAndBiases(
                                                   weightsBk, 
                                                   biasesBk
                                                  );

    // Analytical derivative of the biases
    const Vector dBiases = layers_[layer]->getBiasesDerivative();

    const Scalar error = (dBiases - dBiasesNum).norm() 
                           /  (dBiases.norm() + dBiasesNum.norm());

    errorCheck output;
    output.code = (error < errorLimit) ? EXIT_OK : EXIT_FAIL;
    output.error = error;

    return output;
}

int neuralNetwork::gradientsSanity(
                                           const Matrix& obs, 
                                           const Matrix& labels,
                                           const bool printError
                                          )  const
{    
#ifdef ND_SP
    #error "Single precision used. For testing specify either double or long precision."
#endif

   int code = EXIT_OK;

   forwardPropagation(obs);

   backwardPropagation(obs, labels);

   const Scalar errorLimit = 1.0e-8;
   const Scalar eps = 1.0e-5;

   for(int i = lastLayer_; i >= firstLayer_; --i)
   {
       const errorCheck outputW = 
           checkWeightsGradients(i, obs, labels, eps, errorLimit);

       const errorCheck outputB = 
           checkBiasesGradients(i, obs, labels, eps, errorLimit);

       if(outputW.code == EXIT_FAIL || outputW.code == EXIT_FAIL)
       {           
           code = EXIT_FAIL;
       }

       if(printError || code == EXIT_FAIL)
       {
           std::cout << "Error weights = " 
                       << outputW.error << " " 
                       << ", Error biases = " 
                       << outputB.error 
                       << std::endl;
       }
   }

   return code;
}

} // namespace
