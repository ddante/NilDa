#include <iostream>

#include "conv2DLayer.h"

#include "conv2DUtils.h"

#include "activationFunctions/activationFunctionUtils.h"

#include "activationFunctions/identity.h"
#include "activationFunctions/sigmoid.h"
#include "activationFunctions/relu.h"
#include "activationFunctions/softmax.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

conv2DLayer::conv2DLayer(
                         const int numberOfFilters,
                         const std::array<int, 2>& filterSize,
                         const std::array<int, 2>& filterStride,
                         const bool withPadding,
                         const std::string& activationName
                        ) :
  numberOfFilters_(numberOfFilters),
  filterSize_(filterSize),
  filterStride_(filterStride),
  withPadding_(withPadding),
  activationName_(activationName),
  forwardConvDims_{},
  backwardWeightsConvDims_{}
{
  type_ = layerTypes::conv2D;

  trainable_ = true;
}

conv2DLayer::conv2DLayer(
                         const int numberOfFilters,
                         const std::array<int, 2>& filterSize,
                         const std::array<int, 2>& filterStride,
                         const std::string& activationName
                        ) :
  numberOfFilters_(numberOfFilters),
  filterSize_(filterSize),
  filterStride_(filterStride),
  withPadding_(false),
  activationName_(activationName),
  forwardConvDims_{},
  backwardWeightsConvDims_{}
{
  type_ = layerTypes::conv2D;

  trainable_ = true;
}

conv2DLayer::conv2DLayer(
                         const int numberOfFilters,
                         const std::array<int, 2>& filterSize,
                         const bool withPadding,
                         const std::string& activationName
                        ) :
  numberOfFilters_(numberOfFilters),
  filterSize_(filterSize),
  filterStride_({1,1}),
  withPadding_(withPadding),
  activationName_(activationName),
  forwardConvDims_{},
  backwardWeightsConvDims_{}
{
  type_ = layerTypes::conv2D;

  trainable_ = true;
}

conv2DLayer::conv2DLayer(
                         const int numberOfFilters,
                         const std::array<int, 2>& filterSize,
                         const std::string& activationName
                        ) :
  numberOfFilters_(numberOfFilters),
  filterSize_(filterSize),
  filterStride_({1,1}),
  withPadding_(false),
  activationName_(activationName),
  forwardConvDims_{},
  backwardWeightsConvDims_{}
{
  type_ = layerTypes::conv2D;

  trainable_ = true;
}

void conv2DLayer::init(const layer* previousLayer)
{
  switch (activationFunctionCode(activationName_))
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
  default :
    std::cerr << "Not valid activation function  "
              << activationName_
              << " in this context." << std::endl;
    assert(false);
  }

  if (
      previousLayer->layerType() != layerTypes::input &&
      previousLayer->layerType() != layerTypes::conv2D
     )
  {
    std::cerr << "Previous layer of type "
             <<  layerName(previousLayer->layerType())
             << " not compatible with current layer of type "
             << layerName(type_) << "." << std::endl;

    assert(false);
  }

  const layerSizes prevLayer = previousLayer->size();

  if (prevLayer.isFlat)
  {
    std::cerr << "Previous layer to " << layerName(type_)
              << " cannot be flat." << std::endl;
  }

  assert(prevLayer.rows > 0);
  assert(prevLayer.cols > 0);
  assert(prevLayer.channels > 0);

  const std::array<int, 3> prevLayerSize{
                                         prevLayer.rows,
                                         prevLayer.cols,
                                         prevLayer.channels
                                        };

  setConv2DDims(
                prevLayerSize,
                numberOfFilters_,
                filterSize_,
                filterStride_,
                withPadding_,
                forwardConvDims_,
                backwardWeightsConvDims_,
                backwardInputConvDims_
               );

  size_.isFlat = false;

  size_.size = forwardConvDims_.outputRows
             * forwardConvDims_.outputCols
             * forwardConvDims_.outputChannels;

  size_.rows = forwardConvDims_.outputRows;
  size_.cols = forwardConvDims_.outputCols;
  size_.channels = forwardConvDims_.outputChannels;

  const int kernelDimension = forwardConvDims_.kernelRows
                            * forwardConvDims_.kernelCols
                            * forwardConvDims_.kernelChannels;

  Scalar epsilonInit = sqrt(2.0)/sqrt(kernelDimension);

  const int kernelSize = forwardConvDims_.kernelRows
                       * forwardConvDims_.kernelCols;

  const int kernelChannels = forwardConvDims_.kernelChannels
                           * forwardConvDims_.kernelNumber;

  filterWeights_.setRandom(kernelSize, kernelChannels);

  filterWeights_ *= epsilonInit;

  dFilterWeights_.setZero(kernelSize, kernelChannels);

  biases_.setZero(forwardConvDims_.kernelNumber);

  dBiases_.setZero(forwardConvDims_.kernelNumber);
}

void conv2DLayer::checkInputSize(const Matrix& inputData) const
{
  if (inputData.cols() % forwardConvDims_.inputChannels != 0)
  {
    assert(false);
  }

  if (inputData.rows() !=
      forwardConvDims_.inputRows * forwardConvDims_.inputCols
     )
  {
    assert(false);
  }

}

void conv2DLayer::forwardPropagation(const Matrix& input)
{
#ifdef ND_DEBUG_CHECKS
  checkInputSize(input);
#endif

  const int nObs = input.cols()
                 / forwardConvDims_.inputChannels;

  // Convolve Input with the filters
  convolve(
           nObs,
           forwardConvDims_,
           input.data(),
           filterWeights_.data(),
           linearOutput_
          );

  // Add the biases
  int colObs = 0;

  for (
       int i = 0; i < nObs; ++i,
       colObs += forwardConvDims_.kernelNumber
      )
  {
    linearOutput_.block(
                        0,
                        colObs,
                        linearOutput_.rows(),
                        forwardConvDims_.kernelNumber
                       ).rowwise() += biases_.transpose();
  }

  // Apply the non-linear activation function
  activation_.resize(
                     linearOutput_.rows(),
                     linearOutput_.cols()
                     );

  activationFunction_->applyForward(
                                    linearOutput_,
                                    activation_
                                   );
}


void conv2DLayer::backwardPropagation(
                                      const Matrix& dActivationNex,
                                      const Matrix& inputData
                                     )
{}

void conv2DLayer::setWeightsAndBiases(
                                      const Matrix& W,
                                      const Vector& b
                                     )
{}

void conv2DLayer::incrementWeightsAndBiases(
                                            const Matrix& deltaW,
                                            const Vector& deltaB
                                           )
{}


errorCheck conv2DLayer::localChecks(
                                    const Matrix& input,
                                    const Scalar errorLimit
                                   ) const
{
  checkInputSize(input);

  Scalar error = checkConvolution(
                                  input,
                                  filterWeights_,
                                  forwardConvDims_
                                 );

  errorCheck output;

  output.code = (error < errorLimit) ? EXIT_OK : EXIT_FAIL;
  output.error = error;

  return output;
}


} // namespace
