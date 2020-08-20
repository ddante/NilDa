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
  undoFlattening_(false),
  nObservations_(0),
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
  undoFlattening_(false),
  nObservations_(0),
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
  undoFlattening_(false),
  nObservations_(0),
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
  undoFlattening_(false),
  nObservations_(0),
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
      previousLayer->layerType() != layerTypes::conv2D &&
      previousLayer->layerType() != layerTypes::maxPool2D
     )
  {
    std::cerr << "Previous layer of type "
             <<  getLayerName(previousLayer->layerType())
             << " not compatible with current layer of type "
             << getLayerName(type_) << "." << std::endl;

    assert(false);
  }

  const layerSizes prevLayer = previousLayer->size();

  if (prevLayer.isFlat)
  {
    std::cerr << "Previous layer to " << getLayerName(type_)
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

  biases_.setRandom(forwardConvDims_.kernelNumber);
  biases_ *= epsilonInit;

  dBiases_.setZero(forwardConvDims_.kernelNumber);
}

void conv2DLayer::setupBackward(const layer* nextLayer)
{
  const layerSizes sLayer = nextLayer->size();

  undoFlattening_ = (sLayer.isFlat) ? true : false;
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
  convolve2D(
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

  nObservations_ = nObs;

  //  std::cout << "activation conv2d\n"
  //            << activation_.array() << "\n\n";
}


void conv2DLayer::backwardPropagation(
                                      const Matrix& dActivationNext,
                                      const Matrix& input
                                     )
{
  Matrix dLinearOutput(
                       linearOutput_.rows(),
                       linearOutput_.cols()
                      );

  const int nObs = input.cols()
                 / forwardConvDims_.inputChannels;

  if (undoFlattening_)
  {
    // The next layer is a flatten layer (dense)
    // so the cache must be rearranged in a 2D form
    ConstMapMatrix dActivationNextM(
                                    dActivationNext.data(),
                                    linearOutput_.rows(),
                                    linearOutput_.cols()
                                   );

   activationFunction_->applyBackward(
                                      linearOutput_,
                                      dActivationNextM,
                                      dLinearOutput
                                     );
  }
  else
  {
     // The next layer is a 2d layer
     // no need to rearrange the cache
     activationFunction_->applyBackward(
                                        linearOutput_,
                                        dActivationNext,
                                        dLinearOutput
                                       );
  }

#ifdef ND_DEBUG_CHECKS
    //checkInputAndCacheSize(inputData, dActivationNext);

    assert(nObs == nObservations_);
#endif

  // In the backward convolution, the number of
  // observation and number of input channels are swapped
  backwardWeightsConvDims_.setInputChannels(nObs);

  convolve2D(
             forwardConvDims_.inputChannels,
             backwardWeightsConvDims_,
             input.data(),
             dLinearOutput.data(),
             dFilterWeights_
            );

//  std::cout << "Backward Weights Conv Dim" << std::endl;
//  std::cout << backwardWeightsConvDims_ << std::endl;

  dFilterWeights_ /= nObs;

  // ----------------------------------------------------

#ifdef ND_DEBUG_CHECKS
  const int mappedOutupRows = forwardConvDims_.outputRows
                            * forwardConvDims_.outputCols;

  const int mappedOutupCols = forwardConvDims_.outputChannels
                            * nObs;

  assert(linearOutput_.rows() ==  mappedOutupRows);
  assert(linearOutput_.cols() ==  mappedOutupCols);
#endif

  // The dZ  (output derivative) is first summed over the output size
  // (i.e. linearOutput.Rows), the resulting vector has length
  // linearOutput.Cols (i.e. output.Channels * nObservation).
  // This vector is then arranged as a matrix
  // (Channels , nObservation) and the row summed
  // over the nObservations and then divided by the nObservation
  // (i.e. the mean). Thus, the resulting vector
  // (i.e the derivative of the biases) has size kernel.numberFilters
  ConstMapMatrix mappedOutput(
                              dLinearOutput.data(),
                              linearOutput_.rows(),
                              linearOutput_.cols()
                             );

  Vector sumOutputs = mappedOutput.colwise().sum();

  ConstMapMatrix mappedSumOutputs(
                                  sumOutputs.data(),
                                  forwardConvDims_.outputChannels,
                                  nObs
                                 );

  dBiases_.noalias() = mappedSumOutputs.rowwise().mean();

  // ----------------------------------------------------

  Matrix rotatedKernels;

  applyRotation(
                filterWeights_,
                forwardConvDims_,
                rotatedKernels
               );

  const int cacheRows = forwardConvDims_.inputRows
                      * forwardConvDims_.inputCols;

  const int cacheCols = forwardConvDims_.inputChannels
                      * nObs;

  cacheBackProp_.resize(cacheRows, cacheCols);

  //std::cout << "Backward Input Conv Dim" << std::endl;
  //std::cout << backwardInputConvDims_ << std::endl;

  convolve2D(
             nObs,
             backwardInputConvDims_,
             dLinearOutput.data(),
             rotatedKernels.data(),
             cacheBackProp_
           );
}

void conv2DLayer::setWeightsAndBiases(
                                      const Matrix& W,
                                      const Vector& b
                                     )
{
  if (W.rows() != filterWeights_.rows() ||
      W.cols() != filterWeights_.cols())
  {
    std::cerr << "Size of the input weights matrix "
              << "(" << W.rows() << ", "
              << W.cols() << ") "
              << " not consistent with the layer weights size "
              << "(" << filterWeights_.rows() << ", "
              << filterWeights_.cols() << ") "
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

  filterWeights_.noalias() = W;

  biases_.noalias() = b;
}

void conv2DLayer::incrementWeightsAndBiases(
                                            const Matrix& deltaW,
                                            const Vector& deltaB
                                           )
{
#ifdef ND_DEBUG_CHECKS
  if (deltaW.rows() != filterWeights_.rows() ||
      deltaW.cols() != filterWeights_.cols())
  {
    std::cerr << "Size of the input weights matrix "
              << "(" << deltaW.rows() << ", "
              << deltaW.cols() << ") "
              << " not consistent with the layer weights size "
              << "(" << filterWeights_.rows() << ", "
              << filterWeights_.cols() << ") "
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

  filterWeights_ += deltaW;

  biases_ += deltaB;
}


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
