#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>

#include "primitives/Scalar.h"
#include "layer.h"

namespace NilDa
{


class neuralNetwork
{

 private: 

     std::vector<layer*> layers;

public:

    //neuralNetwork() = delete;

    neuralNetwork(const std::vector<layer*>& vectorLayer):
        layers(vectorLayer)
    {
         // The first layer is the input layer and 
         // it might depend only on the second layer
         //layers[0].init( layers[1] );

         // The hidden and output layers depend on the previous layer
         const int nLayers(layers.size());

         for(int i =1; i < nLayers ; ++i)
         {
             layers[i]->init(layers[i-1]);
         }
    }

    //void addLayer(const layer singleLayer);

    ~neuralNetwork() 
    {
        const int nLayers(layers.size());

        for(int i =1; i < nLayers ; ++i)
        {
             delete layers[i];
        }
    };

};
      

}

#endif