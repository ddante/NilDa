#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include <iostream>
#include <array>

#include "primitives/Vector.h"
#include "primitives/Matrix.h"

#include "layer.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{

class inputLayer : public layer
{

private:    

    // Number of neurons of the layer for a flatten (1D) input
    int inputSize_;

    // Layer sizes for a 2D input with channels
    int inputRows_;
    int inputCols_;
    int inputChannels_;
    
    // Specify if it this a flatten (1D) input layer 
    bool flattenLayer_;

public:

    // Constructors

    inputLayer(const int inSize);

    inputLayer(const std::array<int,3>& inSize);

    // Destructor 

    ~inputLayer() {}

    // Member functions
    void init(const layer* previousLayer) override
    {
        std::cerr << "An input layer cannot call the init." << std::endl;

        assert(false);
    }

    void forwardPropagation(const Matrix& inputData) override
    {
        std::cerr << "An input layer cannot call forwardPropagation." << std::endl;

        assert(false);
    }

    int size() const override 
    {
        return inputSize_;
    }

    void size(std::array<int, 3>& inSizes) const
    {
        inSizes = 
        {
            inputRows_, 
            inputCols_, 
            inputChannels_
        };
    }

};


} // namespace

#endif