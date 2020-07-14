#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>

#include "primitives/Scalar.h"
#include "primitives/Vector.h"
#include "primitives/Matrix.h"

#include "layers/layer.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{


class neuralNetwork
{

 private:

     // List of all the layer in the neural network
     std::vector<layer*> layers_;

     // Total number of layers (input + hidden + output)
     int numberOfLayers_;

public:
    
    // Constructor 
    
    neuralNetwork() = delete;

    neuralNetwork(const std::vector<layer*>& vectorLayer);

    //void addLayer(const layer singleLayer);

    void forwardPropagation(const Matrix& trainingData);

    // Destructor

    ~neuralNetwork() 
    {
        for(int i =0; i < numberOfLayers_; ++i)
        {
             delete layers_[i];
        }
    };

};
      

} // namespace

#endif