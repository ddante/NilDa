#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <iostream>
#include <memory>

#include "primitives/Vector.h"
#include "primitives/Matrix.h"

#include "activationFunction.h"

#include "layer.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{

class denseLayer : public layer
{

private:    

    // Number of neurons of the current layer
    int layerSize_;

    std::unique_ptr<activationFunction> activation;

    // Linear output and activation 
    Matrix linearOutput_;
    Matrix Activation_;

    // Weigh matrix and derivative of the weights
    Matrix Weights_;
    Matrix dWeights_;

    // Bias vector and derivative of the bias 
    Vector biaes_;
    Vector dbiases_;

public:

    // Constructor

    denseLayer(
                    const int inSize, 
                    const std::string activationNam
                   );

    // Destructor 

    //~denseLayer();

    // Member functions

    void init(const layer* previousLayer) override;

    void forwardPropagation(const Matrix& inputData)  override;

    int size() const override 
    {
        return layerSize_;
    }

    void size(std::array<int, 3>& sizes)  const override
    {
         std::cerr << "A dense layer cannot call multi-D size function" << std::endl;

        assert(false);
    }
};


}

#endif