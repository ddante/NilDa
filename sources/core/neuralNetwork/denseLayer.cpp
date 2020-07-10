#include "denseLayer.h"
#include <iostream>

// --------------------------------------------------------------------------- 

namespace NilDa
{


void denseLayer::init(const layer* previousLayer)
{
    assert(
        previousLayer->layerType == INPUT ||
        previousLayer->layerType == DENSE || 
        previousLayer->layerType == DROPOUT || 
        previousLayer->layerType == FLATTEN
    );

    const int prevLayerSize = previousLayer->size();

    Scalar epilonInit = sqrt(6.0)
                           / sqrt(layerSize + prevLayerSize);

    Weights.setRandom(layerSize, prevLayerSize);
    Weights *= epilonInit;
}


}