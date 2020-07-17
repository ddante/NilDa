#ifndef LAYER_H
#define LAYER_H

#include <assert.h>
#include <array>

#include "primitives/Matrix.h"
#include "primitives/Vector.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{

enum class layerTypes {
    input, 
    dense,
    dropout, 
    flatten
};

class layer
{

protected:

    layerTypes type_;

public:

    // Constructors

    layer() {}

    // Member functions

    // Initialize the layer
    virtual void init(const layer* previousLayer) = 0;
   
    // Check if the dimensions of input data are consistent with the layer
    virtual void checkInputSize(const Matrix& inputData) const = 0;

    // Forward step for the layer. 
    // It takes as input the output of the previous layer
    // and computes the activation of the current layer
    virtual void forwardPropagation(const Matrix& inputData)  = 0;

    // Backward step for the layer. 
    // It takes as inputs the back propagation cache of 
    // the next layer and the output of the previous layer. 
    // It computes the derivative of the loss w.r.t. the
    // weights and biases of the current layer and stores
    // the cache to be used by the previous layer
    virtual void backwardPropagation(
                                             const Matrix& dActivationNext, 
                                             const Matrix& inputData
                                            ) = 0;

    virtual const Matrix& getWeights()  const = 0;

    virtual const Vector& getBiases()  const = 0;

    virtual const Matrix& getWeightsDerivative()  const = 0;

    virtual const Vector& getBiasesDerivative()  const = 0;

    virtual const Matrix& output() const = 0;

    virtual const Matrix& backPropCache() const = 0;

    virtual int size() const = 0;

    virtual void size(std::array<int, 3>& sizes) const = 0;

    //virtual void update() = 0;  

    layerTypes layerType()  const
    {
        return type_;
    }

    // Return the string name of the type layer from the enum name
    std::string layerName(const layerTypes inLayerType) const
    {
        std::string name;

        if (inLayerType == layerTypes::input)
        {
            name = "input";
        }
        else if (inLayerType == layerTypes::dense)
        {
            name = "dense";
        }
        else if (inLayerType == layerTypes::dropout)
        {
            name = "drop out";
        }
        else if (inLayerType == layerTypes::flatten)
        {
            name = "flatten";
        }
        else
        {
            std::cerr << "Unknown layer type code." << std::endl;
            assert(false);
        }

        return name;
    }

    // Set the values of the weights and biases
    // in the current layer. Mainly for debugging
    virtual void setWeightsAndBiases(
                                             const Matrix& W, 
                                             const Vector& b
                                            ) = 0;

    // Destructor
    virtual ~layer()  = default;
};


} // namespace

#endif