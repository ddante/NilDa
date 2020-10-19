#include <iostream>
#include <math.h>

#include "denseLayer.h"

#include "activationFunctions/activationFunctionUtils.h"

#include "activationFunctions/identity.h"
#include "activationFunctions/sigmoid.h"
#include "activationFunctions/relu.h"
#include "activationFunctions/softmax.h"

#include "utils/Random.h"
// ---------------------------------------------------------------------------

namespace NilDa
{

denseLayer::denseLayer():
    layerSize_(0),
    activationTypeBK_(activationFunctions::none),
    needFlattening_(false),
    inputSize_(0),
    inputChannels_(0),
    useBiases_(true),
    nObservations_(0)
{
  type_ = layerTypes::dense;

  size_.isFlat = true;
  size_.size = 0;
  size_.rows = 0;
  size_.cols = 0;
  size_.channels = 0;

  activationType_ = activationFunctions::none;

  trainable_ = true;
}

denseLayer::denseLayer(
                       const int inSize,
                       const std::string& activationName
                      ):
    layerSize_(inSize),
    activationTypeBK_(activationFunctions::none),
    needFlattening_(false),
    inputSize_(0),
    inputChannels_(0),
    useBiases_(true),
    nObservations_(0)
{
  type_ = layerTypes::dense;

  size_.isFlat = true;
  size_.size = inSize;
  size_.rows = 0;
  size_.cols = 0;
  size_.channels = 0;

  activationType_ = activationFunctionCode(activationName);

  activationTypeBK_ = activationType_;

  trainable_ = true;

  if (layerSize_ == 1 &&
      activationType_ == activationFunctions::softmax)
  {
    std::cout << "Softmax activation requires layer size > 1.\n";

    std::abort();
  }
}

void denseLayer::checkInput() const
{
  assert(layerSize_ > 0);

  assert(size_.size > 0);
  assert(size_.rows == 0);
  assert(size_.cols == 0);
  assert(size_.channels == 0);
}

void denseLayer::setupForward(const layer* previousLayer)
{
  // Check that the previous layer is compatible
  // with the current layer
  if (
      previousLayer->layerType() != layerTypes::input   &&
      previousLayer->layerType() != layerTypes::dense   &&
      previousLayer->layerType() != layerTypes::dropout &&
      previousLayer->layerType() != layerTypes::conv2D  &&
      previousLayer->layerType() != layerTypes::maxPool2D &&
      previousLayer->layerType() != layerTypes::batchNormalization
     )
  {
    std::cerr << "Previous layer of type "
              <<  getLayerName(previousLayer->layerType())
              << " not compatible with current layer of type "
              << getLayerName(type_) << ".\n";

    std::abort();
  }

  const layerSizes prevLayer = previousLayer->size();

  inputSize_ = prevLayer.size;

  assert(inputSize_ > 0);

  inputChannels_ = 0;

  needFlattening_ = false;

  if (!prevLayer.isFlat)
  {
     needFlattening_ = true;

     inputChannels_ = prevLayer.channels;

     assert(inputChannels_ > 0);
  }
}

void denseLayer::setupBackward(const layer* nextLayer)
{
  if (nextLayer->layerType() == layerTypes::batchNormalization)
  {
    useBiases_ = false;

    // If batchNorm is used, the activation function
    // must not be applied, therefore it is set
    // as the identity
    activationType_ = activationFunctions::identity;
  }
}

void denseLayer::init(const bool resetWeightBiases)
{
  setActivationFunction(activationType_);

  if (resetWeightBiases)
  {
    const Scalar epsilonInit = sqrt(6.0)
                             / sqrt(layerSize_ + inputSize_);

    weights_.setRandom(layerSize_, inputSize_);
    weights_ *= epsilonInit;

    // NOTE: Double check the gradient reset
    // when the restart from a check point is available
    dWeights_.setZero(layerSize_, inputSize_);

    // If the layer is followed by a batchNorm
    // the biases are not needed
    if (useBiases_)
    {
      biases_.setRandom(layerSize_);
      biases_ *= epsilonInit;

      // NOTE: Double check the gradient reset
      // when the restart from a check point is available
      dBiases_.setZero(layerSize_);
    }
  }
}

void denseLayer::checkInputSize(const Matrix& inputData) const
{
  if (weights_.cols() != inputData.rows())
  {
    std::cerr << "Size of input data "
    << "(" << inputData.rows() << ") "
    << " not consistent with dense layer weights size"
    << "(" << weights_.rows() << ", "
    << weights_.cols() << ").\n";

    std::abort();
  }
}

void denseLayer::checkInputAndCacheSize(
                                        const Matrix& inputData,
                                        const Matrix& cacheBackProp
                                       ) const
{
  checkInputSize(inputData);

  if (cacheBackProp.rows() != activation_.rows() &&
      cacheBackProp.cols() != activation_.cols() )
  {
    std::cerr << "Size of the back propagation cache "
    << "(" << cacheBackProp.rows() << ", "
           << cacheBackProp.cols() << ") "
    << " not consistent with the activation size "
    << "(" << activation_.rows() << ", "
           << activation_.cols() << ").\n";

    std::abort();
  }
}

void denseLayer::forwardPropagation(
                                    const Matrix& inputData,
                                    const bool trainingPhase
                                   )
{
  if (needFlattening_)
  {
    // Factor out the channels from the number of colums
    // for a not flat input
    nObservations_ = inputData.cols() / inputChannels_;

    // Constant map to get a flatten the input
    ConstMapMatrix input(
                         inputData.data(),
                         inputSize_,
                         nObservations_
                        );

#ifdef ND_DEBUG_CHECKS
    checkInputSize(input);
#endif

     logit_.resize(weights_.rows(), input.cols());

     // Apply the weights of the layer to the input
     logit_.noalias() = weights_ * input;
  }
  else
  {
#ifdef ND_DEBUG_CHECKS
    checkInputSize(inputData);
#endif

    // For a flat input layer the number of colums is nObs
    nObservations_ = inputData.cols();

    logit_.resize(weights_.rows(), inputData.cols());

    // Apply the weights of the layer to the input
    logit_.noalias() = weights_ * inputData;
  }

  // Add the biases
  if (useBiases_)
  {
    logit_.colwise() += biases_;
  }

  // Apply the activation function
  activation_.resize(
                     logit_.rows(),
                     logit_.cols()
                    );

  activationFunction_->applyForward(
                                    logit_,
                                    activation_
                                   );
}

void denseLayer::backwardPropagation(
                                     const Matrix& dActivationNext,
                                     const Matrix& inputData
                                    )
{
  Matrix dLinearOutput(
                       logit_.rows(),
                       logit_.cols()
                      );

  activationFunction_->applyBackward(
                                     logit_,
                                     dActivationNext,
                                     dLinearOutput
                                    );

  int nObs;

  if (needFlattening_)
  {
    nObs = inputData.cols() / inputChannels_;

    ConstMapMatrix input(
                         inputData.data(),
                         inputSize_,
                         nObservations_
                        );

#ifdef ND_DEBUG_CHECKS
  checkInputAndCacheSize(input, dActivationNext);

  assert(nObs == nObservations_);
#endif

    dWeights_.noalias() = (1.0/nObs)
                        * dLinearOutput
                        * input.transpose();
  }
  else
  {
    nObs = inputData.cols();

#ifdef ND_DEBUG_CHECKS
  checkInputAndCacheSize(inputData, dActivationNext);

  assert(nObs == nObservations_);
#endif

    dWeights_.noalias() = (1.0/nObs)
                        * dLinearOutput
                        * inputData.transpose();
  }

  if (useBiases_)
  {
    dBiases_.noalias() = (1.0/nObs)
                       * dLinearOutput.rowwise().sum();
  }

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
              << weights_.cols() << ").\n";

    std::abort();
  }

  weights_.noalias() = W;

  if (useBiases_)
  {
    if (b.rows() != biases_.rows())
    {
      std::cerr << "Size of the input biases vector "
                << "(" << b.rows() << ") "
                << " not consistent with the layer biases size "
                << "(" << biases_.rows() << ").\n";

      std::abort();
    }

    biases_.noalias() = b;
  }
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
            << weights_.cols() << ").\n";

    std::abort();
  }
#endif

  weights_ += deltaW;

#ifdef ND_DEBUG_CHECKS
  if (useBiases_)
  {
    if (deltaB.rows() != biases_.rows())
    {
      std::cerr << "Size of the input biases vector "
                << "(" << deltaB.rows() << ") "
                << " not consistent with the layer biases size "
                << "(" << biases_.rows() << ").\n";

      std::abort();
    }
  }
#endif

  if (useBiases_)
  {
    biases_ += deltaB;
  }
}

void denseLayer::saveLayer(std::ofstream& ofs) const
{
  const int iType =
    static_cast<
                std::underlying_type_t<layerTypes>
               >(layerTypes::dense);

  ofs.write((char*) (&iType), sizeof(int));

  ofs.write((char*) (&trainable_), sizeof(bool));

  ofs.write((char*) (&size_.size), sizeof(int));

  ofs.write((char*) (&useBiases_), sizeof(bool));

  // Save the backup copy of the activation function type
  const int activationCode =
    static_cast<
                std::underlying_type_t<activationFunctions>
               >(activationTypeBK_);

  ofs.write((char*) (&activationCode), sizeof(int));

  const int wRows = weights_.rows();
  const int wCols = weights_.cols();

  const std::size_t weightsBytes = sizeof(Scalar)
                                 * wRows
                                 * wCols;

  ofs.write((char*) (&wRows), sizeof(int));
  ofs.write((char*) (&wCols), sizeof(int));
  ofs.write((char*) weights_.data(), weightsBytes);

  const int bRows = biases_.rows();

  ofs.write((char*) (&bRows), sizeof(int));

  if (bRows > 0)
  {
    const std::size_t biasesBytes = sizeof(Scalar) * bRows;

    ofs.write((char*) biases_.data(), biasesBytes);
  }
}

void denseLayer::loadLayer(std::ifstream& ifs)
{
  ifs.read((char*) (&trainable_), sizeof(bool));

  ifs.read((char*) (&size_.size), sizeof(int));

  ifs.read((char*) (&useBiases_), sizeof(bool));

  layerSize_ = size_.size;

  size_.rows = 0;
  size_.cols = 0;
  size_.channels = 0;

  int code;
  ifs.read((char*) (&code), sizeof(int));

  activationType_ =
    static_cast<activationFunctions>(code);

  int wRows, wCols;
  ifs.read((char*) (&wRows), sizeof(int));
  ifs.read((char*) (&wCols), sizeof(int));

  weights_.resize(wRows, wCols);

  const std::size_t weightsBytes = sizeof(Scalar) * wRows * wCols;

  ifs.read((char*) weights_.data(), weightsBytes);

  int bRows;
  ifs.read((char*) (&bRows), sizeof(int));

  if (bRows > 0)
  {
    biases_.resize(bRows);

    const std::size_t biasesBytes = sizeof(Scalar) * bRows;

    ifs.read((char*) biases_.data(), biasesBytes);
  }
}

void denseLayer::setActivationFunction(const activationFunctions code)
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
    case activationFunctions::softmax :
      activationFunction_ = std::make_unique<softmax>();
    break;
    default :
       std::cerr << "Not valid activation function.\n";
       std::abort();
  }
}

} // namespace
