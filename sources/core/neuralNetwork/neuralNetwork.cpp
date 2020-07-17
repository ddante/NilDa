#include <iostream>
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
#ifdef NILDA_DEBUG_BUILD
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

} // namespace
