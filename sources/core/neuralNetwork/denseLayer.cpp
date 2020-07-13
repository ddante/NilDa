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
            activation = std::make_unique<identity>();
            break;
        case activationFucntions::sigmoid :
            activation = std::make_unique<sigmoid>();            
            break;
       case activationFucntions::relu :
            activation = std::make_unique<relu>();
            break;            
       case activationFucntions::softmax :
            activation = std::make_unique<softmax>();
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

void denseLayer::forwardPropagation(
                                                 const Matrix& data, 
                                                 const layer* previousLayer
                                                ) 
{

}

} // namespace