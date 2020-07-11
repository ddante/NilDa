#include "denseLayer.h"
#include <iostream>

// --------------------------------------------------------------------------- 

namespace NilDa
{

denseLayer::denseLayer(const int inSize):
    layerSize_(inSize) 
{
    type_ = LAYER_TYPE_DENSE;

    assert(layerSize_ > 0);
}

void denseLayer::init(const layer* previousLayer)
{
    // Check that the previous layer is compatible
    // with the current layer
   if
   (
       previousLayer->layerType() != LAYER_TYPE_INPUT &&
       previousLayer->layerType() != LAYER_TYPE_DENSE && 
       previousLayer->layerType() != LAYER_TYPE_DROPOUT &&
       previousLayer->layerType() != LAYER_TYPE_FLATTEN
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


} // namespace