#include <iostream>
#include "neuralNetwork.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{


neuralNetwork::neuralNetwork(const std::vector<layer*>& vLayers):
    layers_(vLayers),
    numberOfLayers_(vLayers.size())
{
     // The first layer must be an input layer
     if (layers_[0]->layerType() != LAYER_TYPE_INPUT)
     {
         std::cerr << "First layer is not an input layer." << std::endl;
         assert(false);
     }

    // Initialize the hidden and the output layers
    for (int i =1; i < numberOfLayers_; ++i)
    {
        layers_[i]->init(layers_[i-1]);
    }
}


} // namespace
