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

    // Column stride for each observation.
    // It is 1 for flatten layer. 
    // It is the number of channels for 2D input
    int observationStride_; 
    
    // Specify if it this a flatten (1D) input layer 
    bool flattenLayer_;
    
    mutable int numberOfObservations_;

public:

    // Constructors

    inputLayer(const int inSize);

    inputLayer(const std::array<int,3>& inSize);

    // Destructor 

    ~inputLayer()  = default;

    // Member functions
    void init(const layer* previousLayer) override
    {
        std::cerr << "Input layer cannot call the init." << std::endl;

        assert(false);
    }

    void checkInputSize(const Matrix& obs) const override;

    void forwardPropagation(const Matrix& obs) override
    {
        std::cerr << "Input layer cannot call forwardPropagation." << std::endl;

        assert(false);
    }

    void backwardPropagation(
                                    const Matrix& dActivationNext, 
                                    const Matrix& inputData
                                   ) override
    {
        std::cerr << "Input layer cannot call backwardPropagation." << std::endl;

        assert(false);
    }

    const Matrix& getWeights() const override
    {
        std::cerr << "Input layer cannot call getWeights." << std::endl;

        assert(false);
    }

    const Vector& getBiases() const override
    {
        std::cerr << "Input layer cannot call getBiases." << std::endl;

        assert(false);
    }

    const Matrix& getWeightsDerivative() const override
    {
        std::cerr << "Input layer cannot call getWeightsDerivative." << std::endl;

        assert(false);
    }

    const Vector& getBiasesDerivative() const override
    {
        std::cerr << "Input layer cannot call getBiasesDerivative." << std::endl;

        assert(false);
    }
    
    const Matrix& output() const override
    {
        std::cerr << "Input layer cannot call output." << std::endl;

        assert(false);
    }

    const Matrix& backPropCache() const override
    {
        std::cerr << "Input layer cannot call backPropCache." << std::endl;

        assert(false);    
    }

    void setWeightsAndBiases(
                                    const Matrix& W, 
                                    const Vector& b
                                   ) override
    {
        std::cerr << "Input layer cannot call setWeightsAndBiases." << std::endl;

        assert(false);         
    }

    void incrementWeightsAndBiases(
                                            const Matrix& deltaW, 
                                            const Vector& deltaB                                                   
                                           ) override
    {
        std::cerr << "Input layer cannot call incrementWeightsAndBiases." << std::endl;

        assert(false);         
    }

    int size() const override 
    {
        assert(flattenLayer_);

        return inputSize_;
    }

    void size(std::array<int, 3>& inSizes) const override
    {
        assert(!flattenLayer_);

        inSizes = 
        {
            inputRows_, 
            inputCols_, 
            inputChannels_
        };
    }

    int inputStride() const override
    {
        return observationStride_;
    }

};


} // namespace

#endif