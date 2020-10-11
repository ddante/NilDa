#include <iostream>
#include <math.h>

#include "batchNormalizationLayer.h"

#include "activationFunctions/activationFunctionUtils.h"

#include "activationFunctions/identity.h"
#include "activationFunctions/sigmoid.h"
#include "activationFunctions/relu.h"
#include "activationFunctions/softmax.h"

#include "utils/Random.h"
// ---------------------------------------------------------------------------

namespace NilDa
{

batchNormalizationLayer::batchNormalizationLayer()
{
  type_ = layerTypes::batchNormalization;

  size_.isFlat = true;
  size_.size = 0;
  size_.rows = 0;
  size_.cols = 0;
  size_.channels = 0;

  activationType_ = activationFunctions::none;

  trainable_ = true;
}

void
batchNormalizationLayer::setupForward(const layer* previousLayer)
{
  // Check that the previous layer is compatible
  // with the current layer
  if (previousLayer->layerType() != layerTypes::dense)
  {
    std::cerr << "Previous layer of type "
              <<  getLayerName(previousLayer->layerType())
              << " not compatible with current layer of type "
              << getLayerName(type_) << ".\n";

    std::abort();
  }

  // The BN layer get the activation function of the
  // previous layer
  activationType_ = previousLayer->activationType();

  setActivationFunction(activationType_);

  const layerSizes prevLayer = previousLayer->size();

  // Set the size of the BN layer as the one
  // of the previous layer
  size_.isFlat = prevLayer.isFlat;
  size_.size = prevLayer.size;
  size_.rows = prevLayer.rows;
  size_.cols = prevLayer.cols;
  size_.channels = prevLayer.channels;
}

void
batchNormalizationLayer::init(const bool resetWeightBiases)
{
  if (resetWeightBiases)
  {
    const Scalar epsilonInit = sqrt(2.0)
                             / sqrt(size_.size);

    // For batch norm: Z = gamma*Z_norm + beta
    // weights_ = {gamma}, is a vector
    // biases_  = {beta}
    weights_.setRandom(size_.size, 1);
    weights_ *= epsilonInit;

    biases_.setRandom(size_.size);
    biases_ *= epsilonInit;

    dWeights_.setZero(
                      weights_.rows(),
                      weights_.cols()
                     );

    dBiases_.setZero(biases_.rows());
  }
}

void
batchNormalizationLayer::forwardPropagation(
                                            const Matrix& inputData,
                                            const bool trainingPhase
                                           )
{
  const Vector batchMean = inputData.rowwise().mean();

  const Matrix diffSquared = (
                              inputData.colwise() - batchMean
                             ).array().square();

  const Vector batchVar = (
                           diffSquared.rowwise().mean().array()
                           + epsilon_
                          ).array().sqrt();

  Matrix logitNorm = (inputData.colwise() - batchMean);

  logitNorm.array().colwise() /= batchVar.array();

  // Map the column matrix of the weights into a vector;
  ConstMapVector gamma(
                       weights_.data(),
                       weights_.rows()
                      );

  logit_ = logitNorm.array().colwise() * gamma.array();

  logit_.colwise() += biases_;

  activation_.resize(
                     logit_.rows(),
                     logit_.cols()
                    );

  activationFunction_->applyForward(
                                    logit_,
                                    activation_
                                   );
}

void
batchNormalizationLayer::backwardPropagation(
                                             const Matrix& dActivationNext,
                                             const Matrix& inputData
                                            )
{}

void
batchNormalizationLayer::setWeightsAndBiases(
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
}

void
batchNormalizationLayer::incrementWeightsAndBiases(
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
}

void batchNormalizationLayer::saveLayer(std::ofstream& ofs) const
{}

void batchNormalizationLayer::loadLayer(std::ifstream& ifs)
{}

void
batchNormalizationLayer::setActivationFunction(const activationFunctions code)
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
