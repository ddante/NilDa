#include <iostream>

#include "denseLayer.h"

#include "activationFunctions/activationFunctionUtils.h"

#include "activationFunctions/identity.h"
#include "activationFunctions/sigmoid.h"
#include "activationFunctions/relu.h"
#include "activationFunctions/softmax.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{
denseLayer::denseLayer(
                               const int inSize, 
                               const std::string activationName
                              ):
    layerSize_(inSize)
{
    
    type_ = layerTypes::dense;

    switch (activationFunctionCode(activationName))
    {
        case activationFucntions::identity :
            activationFunction_ = std::make_unique<identity>();
            break;
        case activationFucntions::sigmoid :
            activationFunction_ = std::make_unique<sigmoid>();            
            break;
       case activationFucntions::relu :
            activationFunction_ = std::make_unique<relu>();
            break; 
       case activationFucntions::softmax :
            activationFunction_ = std::make_unique<softmax>();
            break; 
       default :
           std::cerr << "Not valid activation function  " 
                        << activationName
                        << " in this context." << std::endl;
           assert(false);
    }
    
    assert(layerSize_ > 0);
}

void denseLayer::init(const layer* previousLayer)
{
    // Check that the previous layer is compatible
    // with the current layer
   if
   (
       previousLayer->layerType() != layerTypes::input &&
       previousLayer->layerType() != layerTypes::dense && 
       previousLayer->layerType() != layerTypes::dropout &&
       previousLayer->layerType() != layerTypes::flatten
   )
   {
         std::cerr << "Previous layer of type " 
                     <<  layerName(previousLayer->layerType())
                     << " not compatible with current layer of type "
                     << layerName(type_) << "." << std::endl;

         assert(false);
    }

    const int prevLayerSize = previousLayer->size();

    Scalar epilonInit = sqrt(6.0)
                           / sqrt(layerSize_ + prevLayerSize);

    Weights_.setRandom(layerSize_, prevLayerSize);
    Weights_ *= epilonInit;

    dWeights_.setZero(layerSize_, prevLayerSize);

    biaes_.setZero(layerSize_);

    dbiases_.setZero(layerSize_);
}

void denseLayer::checkInputSize(const Matrix& inputData)
{
    if (layerSize_ != inputData.rows())
    {
        std::cerr << "Size of the input data "
        << "(" << inputData.rows() << ") "
        << " not consistent with the input layer size" 
        << "(" << layerSize_ << ") "
        << std::endl;

        assert(false);
    }
}

void denseLayer::forwardPropagation(const Matrix& inputData) 
{    
#ifdef NILDA_DEBUG_BUILD
    checkInputSize(inputData);
#endif

    linearOutput_.resize(
                                Weights_.rows(), 
                                inputData.cols()
                               );

    // Apply the weights of the layer to the input
    linearOutput_.noalias() = Weights_ * inputData;

    std::cout <<"input:\n";
    std::cout << inputData << "\n--------\n";

    std::cout <<"weights:\n";
    std::cout << Weights_ << "\n--------\n";

    std::cout <<"lin output:\n";
    std::cout << linearOutput_ << "\n--------\n";

    // Add the biases 
    linearOutput_.colwise() += biaes_;

    activation_.resize(
                            linearOutput_.rows(), 
                            linearOutput_.cols()
                           );

    // Apply the activation function
    activationFunction_->applyForward(
                                                 linearOutput_, 
                                                 activation_
                                                );
    std::cout <<"activation:\n";
    std::cout << activation_ << "\n--------\n";

}

} // namespace