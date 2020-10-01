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
conv2DLayer::conv2DLayer() :
  numberOfFilters_(0),
  filterSize_{},
  filterStride_{},
  withPadding_(false),
  undoFlattening_(false),
  useBatchNormalization(false),
  nObservations_(0),
  forwardConvDims_{},
  backwardWeightsConvDims_{}
{
  type_ = layerTypes::conv2D;

  activationType_ = activationFunctions::none;

  trainable_ = true;
}

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
  undoFlattening_(false),
  useBatchNormalization(false),
  nObservations_(0),
  forwardConvDims_{},
  backwardWeightsConvDims_{}
{
  type_ = layerTypes::conv2D;

  activationType_ = activationFunctionCode(activationName);

  trainable_ = true;

  setActivationFunction(activationType_);
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
  undoFlattening_(false),
  useBatchNormalization(false),
  nObservations_(0),
  forwardConvDims_{},
  backwardWeightsConvDims_{}
{
  type_ = layerTypes::conv2D;

  activationType_ = activationFunctionCode(activationName);

  trainable_ = true;

  setActivationFunction(activationType_);
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
  undoFlattening_(false),
  useBatchNormalization(false),
  nObservations_(0),
  forwardConvDims_{},
  backwardWeightsConvDims_{}
{
  type_ = layerTypes::conv2D;

  activationType_ = activationFunctionCode(activationName);

  trainable_ = true;

  setActivationFunction(activationType_);
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
  undoFlattening_(false),
  useBatchNormalization(false),
  nObservations_(0),
  forwardConvDims_{},
  backwardWeightsConvDims_{}
{
  type_ = layerTypes::conv2D;

  activationType_ = activationFunctionCode(activationName);

  trainable_ = true;

  setActivationFunction(activationType_);
}

void conv2DLayer::checkInput() const
{
  assert(numberOfFilters_ > 0);

  assert(filterSize_[0] > 0);
  assert(filterSize_[1] > 0);

  assert(filterStride_[0] > 0);
  assert(filterStride_[1] > 0);
}

void conv2DLayer::init(
                       const layer* previousLayer,
                       const bool resetWeightBiases
                      )
{
  if (
      previousLayer->layerType() != layerTypes::input     &&
      previousLayer->layerType() != layerTypes::conv2D    &&
      previousLayer->layerType() != layerTypes::maxPool2D &&
      previousLayer->layerType() != layerTypes::dropout
     )
  {
    std::cerr << "Previous layer of type "
             <<  getLayerName(previousLayer->layerType())
             << " not compatible with current layer of type "
             << getLayerName(type_) << ".\n";

    assert(false);
  }

  const layerSizes prevLayer = previousLayer->size();

  if (prevLayer.isFlat)
  {
    std::cerr << "Previous layer to " << getLayerName(type_)
              << " cannot be flat.\n";

    std::abort();
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

  const int kernelSize = forwardConvDims_.kernelRows
                       * forwardConvDims_.kernelCols;

  const int kernelChannels = forwardConvDims_.kernelChannels
                           * forwardConvDims_.kernelNumber;

  if (resetWeightBiases)
  {
    Scalar epsilonInit = sqrt(2.0)/sqrt(kernelDimension);

    filterWeights_.setRandom(kernelSize, kernelChannels);

    filterWeights_ *= epsilonInit;

    biases_.setRandom(forwardConvDims_.kernelNumber);
    biases_ *= epsilonInit;
  }

  dFilterWeights_.setZero(kernelSize, kernelChannels);

  dBiases_.setZero(forwardConvDims_.kernelNumber);
}

void conv2DLayer::setupBackward(const layer* nextLayer)
{
  const layerSizes sLayer = nextLayer->size();

  undoFlattening_ = (sLayer.isFlat) ? true : false;

  if (nextLayer->layerType() == layerTypes::batchNormalization)
  {
    useBatchNormalization = true;
  }
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

void conv2DLayer::forwardPropagation(
                                     const Matrix& input,
                                     const bool trainingPhase
                                    )
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
             logit_
            );

  // Add the biases
  int colObs = 0;

  for (
       int i = 0; i < nObs; ++i,
       colObs += forwardConvDims_.kernelNumber
      )
  {
    logit_.block(
                 0,
                 colObs,
                 logit_.rows(),
                 forwardConvDims_.kernelNumber
                ).rowwise() += biases_.transpose();
  }

  // Apply the non-linear activation function
  activation_.resize(
                     logit_.rows(),
                     logit_.cols()
                     );

  activationFunction_->applyForward(
                                    logit_,
                                    activation_
                                   );

  nObservations_ = nObs;
}


void conv2DLayer::backwardPropagation(
                                      const Matrix& dActivationNext,
                                      const Matrix& input
                                     )
{
  Matrix dLogit(
                       logit_.rows(),
                       logit_.cols()
                      );

  const int nObs = input.cols()
                 / forwardConvDims_.inputChannels;

  if (undoFlattening_)
  {
    // The next layer is a flatten layer (dense)
    // so the cache must be rearranged in a 2D form
    ConstMapMatrix dActivationNextM(
                                    dActivationNext.data(),
                                    logit_.rows(),
                                    logit_.cols()
                                   );

   activationFunction_->applyBackward(
                                      logit_,
                                      dActivationNextM,
                                      dLogit
                                     );
  }
  else
  {
     // The next layer is a 2d layer
     // no need to rearrange the cache
     activationFunction_->applyBackward(
                                        logit_,
                                        dActivationNext,
                                        dLogit
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
             dLogit.data(),
             dFilterWeights_
            );

  dFilterWeights_ /= nObs;

  // ----------------------------------------------------

#ifdef ND_DEBUG_CHECKS
  const int mappedOutupRows = forwardConvDims_.outputRows
                            * forwardConvDims_.outputCols;

  const int mappedOutupCols = forwardConvDims_.outputChannels
                            * nObs;

  assert(logit_.rows() ==  mappedOutupRows);
  assert(logit_.cols() ==  mappedOutupCols);
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
                              dLogit.data(),
                              logit_.rows(),
                              logit_.cols()
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

  convolve2D(
             nObs,
             backwardInputConvDims_,
             dLogit.data(),
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
              << filterWeights_.cols() << ").\n";

    assert(false);
  }

  if (b.rows() != biases_.rows())
  {
    std::cerr << "Size of the input biases vector "
              << "(" << b.rows() << ") "
              << " not consistent with the layer biases size "
              << "(" << biases_.rows() << ").\n";

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
              << filterWeights_.cols() << ").\n";

    assert(false);
  }

  if (deltaB.rows() != biases_.rows())
  {
    std::cerr << "Size of the input biases vector "
              << "(" << deltaB.rows() << ") "
              << " not consistent with the layer biases size "
              << "(" << biases_.rows() << ").\n";

    assert(false);
  }
#endif

  filterWeights_ += deltaW;

  biases_ += deltaB;
}

void conv2DLayer::saveLayer(std::ofstream& ofs) const
{
  const int iType =
    static_cast<std::underlying_type_t<layerTypes> >(layerTypes::conv2D);

  ofs.write((char*) (&iType),      sizeof(int));
  ofs.write((char*) (&trainable_), sizeof(bool));

  ofs.write((char*) (&(forwardConvDims_.outputChannels)), sizeof(int));

  ofs.write((char*) (&(forwardConvDims_.kernelRows)), sizeof(int));
  ofs.write((char*) (&(forwardConvDims_.kernelCols)), sizeof(int));

  ofs.write((char*) (&(forwardConvDims_.kernelStrideRow)), sizeof(int));
  ofs.write((char*) (&(forwardConvDims_.kernelStrideCol)), sizeof(int));

  ofs.write((char*) (&(forwardConvDims_.padding)), sizeof(bool));

  const int activationCode = activationFunction_->type();
  ofs.write((char*) (&activationCode), sizeof(int));

  const int wRows = filterWeights_.rows();
  const int wCols = filterWeights_.cols();

  const std::size_t weightsBytes = sizeof(Scalar)
                                 * wRows
                                 * wCols;

  ofs.write((char*) (&wRows), sizeof(int));
  ofs.write((char*) (&wCols), sizeof(int));
  ofs.write((char*) filterWeights_.data(), weightsBytes);

  const int bRows = biases_.rows();

  const std::size_t biasesBytes = sizeof(Scalar) * bRows;

  ofs.write((char*) (&bRows), sizeof(int));
  ofs.write((char*) biases_.data(), biasesBytes);
}

void conv2DLayer::loadLayer(std::ifstream& ifs)
{
  ifs.read((char*) (&trainable_), sizeof(bool));

  ifs.read((char*) (&(numberOfFilters_)), sizeof(int));

  ifs.read((char*) (&(filterSize_[0])), sizeof(int));
  ifs.read((char*) (&(filterSize_[1])), sizeof(int));

  ifs.read((char*) (&(filterStride_[0])), sizeof(int));
  ifs.read((char*) (&(filterStride_[1])), sizeof(int));

  ifs.read((char*) (&withPadding_), sizeof(bool));

  int code;
  ifs.read((char*) (&code), sizeof(int));

  activationFunctions aType =
    static_cast<activationFunctions>(code);

  setActivationFunction(aType);

  int wRows, wCols;
  ifs.read((char*) (&wRows), sizeof(int));
  ifs.read((char*) (&wCols), sizeof(int));

  filterWeights_.resize(wRows, wCols);

  const std::size_t weightsBytes = sizeof(Scalar) * wRows * wCols;
  ifs.read((char*) filterWeights_.data(), weightsBytes);

  int bRows;
  ifs.read((char*) (&bRows), sizeof(int));

  biases_.resize(bRows);

  const std::size_t biasesBytes = sizeof(Scalar) * bRows;
  ifs.read((char*) biases_.data(), biasesBytes);
}

void conv2DLayer::setActivationFunction(const activationFunctions code)
{
  switch (code)
  {
    case activationFunctions::identity :
      activationFunction_ = std::make_unique<identity>();
    break;
    case activationFunctions::sigmoid :
      activationFunction_ = std::make_unique<sigmoid>();
    break;
    case activationFunctions::relu :
      activationFunction_ = std::make_unique<relu>();
    break;
    default :
       std::cerr << "Not valid activation function.\n";
       assert(false);
  }
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
