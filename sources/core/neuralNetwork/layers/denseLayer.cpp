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
                               const std::string& activationName
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

    const Scalar epilonInit = sqrt(6.0)
                                 / sqrt(layerSize_ + prevLayerSize);

    weights_.setRandom(layerSize_, prevLayerSize);
    weights_ *= epilonInit;

    dWeights_.setZero(layerSize_, prevLayerSize);

    biases_.setZero(layerSize_);

    dBiases_.setZero(layerSize_);
}

void denseLayer::checkInputSize(const Matrix& inputData) const
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

void denseLayer::checkInputAndCacheSize(
                                                     const Matrix& inputData,
                                                     const Matrix& cacheBackProp
                                                    ) const
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

    if(cacheBackProp.rows() != activation_.rows() &&
       cacheBackProp.cols() != activation_.cols() )
    {
        std::cerr << "Size of the back propagation cache "
        << "(" << cacheBackProp.rows() << ", "
                  << cacheBackProp.cols() << ") " 
        << " not consistent with the activation size " 
        << "(" << activation_.rows() << ", "
                  << activation_.cols() << ") " 
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
                               weights_.rows(), 
                               inputData.cols()
                              );

    // Apply the weights of the layer to the input
    linearOutput_.noalias() = weights_ * inputData;

    // Add the biases 
    linearOutput_.colwise() += biases_;

    // Apply the activation function
    activation_.resize(
                            linearOutput_.rows(), 
                            linearOutput_.cols()
                           );

    activationFunction_->applyForward(
                                                 linearOutput_, 
                                                 activation_
                                                );
}

void denseLayer::backwardPropagation(
                                                const Matrix& dActivationNext, 
                                                const Matrix& inputData
                                               )
{
#ifdef NILDA_DEBUG_BUILD
    checkInputAndCacheSize(inputData, dActivationNext);
#endif  

    Matrix dLinearOutput(
                               linearOutput_.rows(), 
                               linearOutput_.cols()
                              );

    activationFunction_->applyBackward(
                                                  linearOutput_, 
                                                  dActivationNext,
                                                  dLinearOutput
                                                 );

    const int nObs = activation_.cols();

    dWeights_.noalias() = (1.0/nObs) 
                              * dLinearOutput 
                              * inputData.transpose();

    dBiases_.noalias() = (1.0/nObs) 
                             * dLinearOutput.rowwise().sum();

    cacheBackProp_.resize(
                                 dWeights_.cols(), 
                                 dLinearOutput.cols()
                                );

    cacheBackProp_.noalias() = weights_.transpose() 
                                     * dLinearOutput;
}


} // namespace