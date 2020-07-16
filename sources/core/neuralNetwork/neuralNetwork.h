#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <memory>

#include "primitives/Scalar.h"
#include "primitives/Vector.h"
#include "primitives/Matrix.h"

#include "layers/layer.h"

#include "lossFunctions/lossFunction.h"

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

     // Store for simplicity the number of the last layer: 
     // numberOfLayers_ - 1
     int lastLayer_;

     mutable bool validState_;

     std::unique_ptr<lossFunction> lossFunction_;

public:
    
    // Constructor 
    
    neuralNetwork() = delete;

    neuralNetwork(const std::vector<layer*>& vectorLayer);

    //void addLayer(const layer singleLayer);

    void forwardPropagation(const Matrix& obs) const;

    void setLossFunction(const std::string& lossName);

    Scalar getLoss(const Matrix& obs, const Matrix& labels) const;

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