#include <iostream>

#include "denseLayer.h"

#include "activationFunctionUtils.h"

#include "identity.h"
#include "sigmoid.h"
#include "relu.h"
#include "softmax.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{
denseLayer::denseLayer(
                                 const int inSize, 
                                 const std::string activationName):
    layerSize_(inSize)
{
    
    type_ = layerTypes::dense;

    switch (activationFunctionCode(activationName))
    {
        case activationFucntions::identity :
            activationFunction_ = std::make_shared<identity>();
            break;
        case activationFucntions::sigmoid :
            activationFunction_ = std::make_shared<sigmoid>();            
            break;
       case activationFucntions::relu :
            activationFunction_ = std::make_shared<relu>();
            break;            
       case activationFucntions::softmax :
            activationFunction_ = std::make_shared<softmax>();
            break; 
       default :
           std::cerr << "Unknown activation function name " 
                       << activationName
                       << "." << std::endl;
           assert(false);
    }
    
    assert(layerSize_ > 0);
}

void denseLayer::init(const layer* previousLayer)
{
    // Check that the previous layer is compatible
    // with the current layer
   if
   (
       previousLayer->layerType() != layerTypes::input &&
       previousLayer->layerType() != layerTypes::dense && 
       previousLayer->layerType() != layerTypes::dropout &&
       previousLayer->layerType() != layerTypes::flatten
   )
   {
         std::cerr << "Previous layer of type " 
                     <<  layerName(previousLayer->layerType())
                     << " not compatible with current layer of type "
                     << layerName(type_) << "." << std::endl;

         assert(false);
    }

    const int prevLayerSize = previousLayer->size();

    Scalar epilonInit = sqrt(6.0)
                           / sqrt(layerSize_ + prevLayerSize);

    Weights_.setRandom(layerSize_, prevLayerSize);
    Weights_ *= epilonInit;

    dWeights_.setZero(layerSize_, prevLayerSize);

    biaes_.setZero(layerSize_);

    dbiases_.setZero(layerSize_);
}

void denseLayer::checkInputSize(const Matrix& inputData)
{
    if (layerSize_ != inputData.rows())
    {
        std::cerr << "Size of the input data "
        << "(" << inputData.rows() << ") "
        << " not consistent with the input layer size" 
        << "(" << layerSize_ << ") "
        << std::endl;

        assert(false);
    }
}

void denseLayer::forwardPropagation(const Matrix& inputData) 
{    
    linearOutput_.resize(
                                Weights_.rows(), 
                                inputData.cols()
                               );

    // Apply the weights of the layer to the input
    linearOutput_.noalias() = Weights_ * inputData;

    // Add the biases 
    linearOutput_.colwise() += biaes_;

    activation_.resize(
                             linearOutput_.rows(), 
                             linearOutput_.cols()
                            );

    // Apply the activation function
    activation_ = 
        activationFunction_->applyForward(linearOutput_);
}

} // namespace