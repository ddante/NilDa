#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <iostream>

#include "primitives/Vector.h"
#include "primitives/Matrix.h"

#include "layer.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{

class denseLayer: public layer
{

private:    

    // Number of neurons of the current layer
    int layerSize;

private:

    // Linear output and activation 
    Matrix linearOutput;
    Matrix Activation;

    // Weigh matrix and derivative of the weights
    Matrix Weights;
    Matrix dWeights;

    // Bias vector and derivative of the bias 
    Vector biaes;
    Vector dbiases;

public:

    // Constructor

    denseLayer(const int inSize):
        layerSize(inSize)       
    {
        layerType = DENSE;
        assert(inSize > 0);
    }

    // Destructor 

    ~denseLayer() {}

    // Member functions

    void init(const layer* previousLayer) override;
    
    int size() const override 
    {
        return layerSize;
    }

};


}

#endif