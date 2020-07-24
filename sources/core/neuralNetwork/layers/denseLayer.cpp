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
    
    trainable_ = true;

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

    if (layerSize_ == 1 && 
        activationFunctionCode(activationName) ==
        activationFucntions::softmax)
    {
        std::cout << "Softmax activation requires layer size > 1" << std::endl;
        assert(false);
    }
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
    if (weights_.cols() != inputData.rows())
    {
        std::cerr << "Size of input data "
        << "(" << inputData.rows() << ") "
        << " not consistent with dense layer weights size" 
        << "(" << weights_.rows() << ", "
        << weights_.cols() << ") "
        << std::endl;

        assert(false);
    }
}

void denseLayer::checkInputAndCacheSize(
                                                     const Matrix& inputData,
                                                     const Matrix& cacheBackProp
                                                    ) const
{
    if (weights_.cols() != inputData.rows())
    {
        std::cerr << "Size of input data "
        << "(" << inputData.rows() << ") "
        << " not consistent with dense layer size" 
        << " not consistent with dense layer weights size" 
        << "(" << weights_.rows() << ", "
        << weights_.cols() << ") "
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
#ifdef ND_DEBUG_CHECKS
    checkInputSize(inputData);
#endif

    nObservations_ = inputData.cols();

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
#ifdef ND_DEBUG_CHECKS
    checkInputAndCacheSize(inputData, dActivationNext);

    assert(inputData.cols() == nObservations_);
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

void denseLayer::setWeightsAndBiases(
                                                const Matrix& W, 
                                                const Vector& b
                                               ) 
{       
    if (W.rows() != weights_.rows() ||
        W.cols() != weights_.cols())
    {
        std::cerr << "Size of the input weights matrix "
        << "(" << W.rows() << ", "
                  << W.cols() << ") " 
        << " not consistent with the layer weights size " 
        << "(" << weights_.rows() << ", "
                  << weights_.cols() << ") " 
        << std::endl;

        assert(false);
    }

    if (b.rows() != biases_.rows())
    {
        std::cerr << "Size of the input biases vector "
        << "(" << b.rows() << ") "
        << " not consistent with the layer biases size " 
        << "(" << biases_.rows() << ") "
        << std::endl;
        
        assert(false);
    }

    weights_.noalias() = W;

    biases_.noalias() = b;
}

void denseLayer::incrementWeightsAndBiases(
                                                        const Matrix& deltaW, 
                                                        const Vector& deltaB                                                   
                                                       )
{
#ifdef ND_DEBUG_CHECKS
    if (deltaW.rows() != weights_.rows() ||
        deltaW.cols() != weights_.cols())
    {
        std::cerr << "Size of the input weights matrix "
        << "(" << deltaW.rows() << ", "
                  << deltaW.cols() << ") " 
        << " not consistent with the layer weights size " 
        << "(" << weights_.rows() << ", "
                  << weights_.cols() << ") " 
        << std::endl;

        assert(false);
    }

    if (deltaB.rows() != biases_.rows())
    {
        std::cerr << "Size of the input biases vector "
        << "(" << deltaB.rows() << ") "
        << " not consistent with the layer biases size " 
        << "(" << biases_.rows() << ") "
        << std::endl;
        
        assert(false);
    }
#endif

    weights_ += deltaW;

    biases_ += deltaB;
}


} // namespace