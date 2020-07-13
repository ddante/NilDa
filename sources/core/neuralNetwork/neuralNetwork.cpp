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
     if (layers_[0]->layerType() != layerTypes::input)
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

void neuralNetwork::forwardPropagation(const Matrix& trainingData)
{
    
    // The first layer is just the input layer: nothing to do

    // The second layer takes in directly the input data
    layers_[1]->forwardPropagation(trainingData);

    for (int i =2; i < numberOfLayers_; ++i)
    {
        // The other layers take in the output of the previous layer
        layers_[i]->forwardPropagation(layers_[i-1]->output());
    }
}


} // namespace
