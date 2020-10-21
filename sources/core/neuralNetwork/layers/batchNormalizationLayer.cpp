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

batchNormalizationLayer::batchNormalizationLayer():
  momentum_(0.9),
  epsilonTol_(1e-10),
  nObservations_(0)
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

batchNormalizationLayer::batchNormalizationLayer(const Scalar momentum):
  momentum_(momentum),
  epsilonTol_(1e-10),
  nObservations_(0)
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

batchNormalizationLayer::batchNormalizationLayer(
                                                 const Scalar momentum,
                                                 const Scalar tol
                                               ):
  momentum_(momentum),
  epsilonTol_(tol),
  nObservations_(0)
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

  const layerSizes prevLayer = previousLayer->size();

  // Set the size of the BN layer as the one
  // of the previous layer
  size_.size = prevLayer.size;
}

void
batchNormalizationLayer::init(const bool resetWeightBiases)
{
  setActivationFunction(activationType_);

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

    // Reset also the drunning mean and variacne
    runningMean_.setZero(size_.size);
    runningVariance_.setZero(size_.size);
  }
}

void
batchNormalizationLayer::checkInputSize(const Matrix& inputData) const
{
  if (weights_.rows() != inputData.rows())
  {
    std::cerr << "Size of input data "
    << "(" << inputData.rows() << ") "
    << " not consistent with batchNorm layer weights size"
    << "(" << weights_.rows() << ", "
    << weights_.rows() << ").\n";

    std::abort();
  }
}

void
batchNormalizationLayer::checkInputAndCacheSize(
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

void
batchNormalizationLayer::forwardPropagation(
                                            const Matrix& inputData,
                                            const bool trainingPhase
                                           )
{
#ifdef ND_DEBUG_CHECKS
    checkInputSize(inputData);
#endif

  nObservations_ = inputData.cols();

  batchMean_ = inputData.rowwise().mean();

  const Matrix diffSquared = (
                              inputData.colwise() - batchMean_
                             ).array().square();

  batchVariance_ = diffSquared.rowwise().mean().array();
                 + epsilonTol_;

  Matrix normLogit;

  if (trainingPhase)
  {
    normLogit = inputData.colwise() - batchMean_;

    normLogit.array().colwise() *= batchVariance_.array().rsqrt();
  }
  else
  {
    normLogit = inputData.colwise() - runningMean_;

    normLogit.array().colwise() *= runningVariance_.array().rsqrt();
  }

  // Map the column matrix of the weights into a vector;
  ConstMapVector gamma(
                       weights_.data(),
                       weights_.rows()
                      );

  logit_= normLogit.array().colwise() * gamma.array();

  logit_.colwise() += biases_;

  activation_.resize(
                     logit_.rows(),
                     logit_.cols()
                    );

  activationFunction_->applyForward(
                                    logit_,
                                    activation_
                                   );
  if (trainingPhase)
  {
    runningMean_ =      momentum_  * runningMean_
                 + (1 - momentum_) * batchMean_;

    runningVariance_ =      momentum_  * runningVariance_
                     + (1 - momentum_) * batchVariance_;
  }
}

void
batchNormalizationLayer::backwardPropagation(
                                             const Matrix& dActivationNext,
                                             const Matrix& inputData
                                            )
{
  const int nObs = inputData.cols();

#ifdef ND_DEBUG_CHECKS
  checkInputAndCacheSize(inputData, dActivationNext);

  assert(nObs == nObservations_);
#endif

  Matrix dLogit(logit_.rows(), logit_.cols());

  activationFunction_->applyBackward(
                                     logit_,
                                     dActivationNext,
                                     dLogit
                                    );

  // First comput logit_norm
  const Matrix zeroMean = inputData.colwise() - batchMean_;

  const Matrix normLogit = zeroMean.array().colwise()
                         * batchVariance_.array().rsqrt();

  const Matrix prod = dLogit.array() * normLogit.array();

  dWeights_.noalias() = (1.0/nObs)
                      * prod.rowwise().sum();

  dBiases_.noalias() = (1.0/nObs)
                     * dLogit.rowwise().sum();

  const Vector prodShift = (
                            dLogit.array() * zeroMean.array()
                           ).rowwise().sum();

  cacheBackProp_ = zeroMean.array().colwise()
                 * (prodShift.array() / batchVariance_.array());

  cacheBackProp_ = cacheBackProp_.colwise()
                 + dLogit.rowwise().sum();

  cacheBackProp_ -= nObs * dLogit;

  // Map the column matrix of the weights into a vector;
  ConstMapVector gamma(
                       weights_.data(),
                       weights_.rows()
                      );

  cacheBackProp_ = cacheBackProp_.array().colwise()
                 * (gamma.array() * batchVariance_.array().rsqrt());

  cacheBackProp_ *= -1.0/nObs;
}

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

  if (b.rows() != biases_.rows())
  {
    std::cerr << "Size of the input biases vector "
              << "(" << b.rows() << ") "
              << " not consistent with the layer biases size "
              << "(" << biases_.rows() << ").\n";

    std::abort();
  }

  weights_.noalias() = W;

  biases_.noalias() = b;
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

  if (deltaB.rows() != biases_.rows())
  {
    std::cerr << "Size of the input biases vector "
              << "(" << deltaB.rows() << ") "
              << " not consistent with the layer biases size "
              << "(" << biases_.rows() << ").\n";

    std::abort();
  }
#endif

  weights_ += deltaW;

  biases_ += deltaB;
}

void batchNormalizationLayer::saveLayer(std::ofstream& ofs) const
{
  const int iType =
    static_cast<
                std::underlying_type_t<layerTypes>
               >(layerTypes::batchNormalization);

  ofs.write((char*) (&iType), sizeof(int));

  ofs.write((char*) (&trainable_), sizeof(bool));

  ofs.write((char*) (&size_.size), sizeof(int));

  ofs.write((char*) (&momentum_), sizeof(Scalar));

  ofs.write((char*) (&epsilonTol_), sizeof(Scalar));

  const int activationCode = activationFunction_->type();

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

  const std::size_t biasesBytes = sizeof(Scalar) * bRows;

  ofs.write((char*) (&bRows), sizeof(int));
  ofs.write((char*) biases_.data(), biasesBytes);

  const int meanRows = runningMean_.rows();

  const std::size_t meanBytes = sizeof(Scalar) * meanRows;

  ofs.write((char*) (&meanRows), sizeof(int));
  ofs.write((char*) runningMean_.data(), meanBytes);
  ofs.write((char*) runningVariance_.data(), meanBytes);
}

void batchNormalizationLayer::loadLayer(std::ifstream& ifs)
{
  ifs.read((char*) (&trainable_), sizeof(bool));

  ifs.read((char*) (&size_.size), sizeof(int));

  ifs.read((char*) (&momentum_), sizeof(Scalar));

  ifs.read((char*) (&epsilonTol_), sizeof(Scalar));

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

  biases_.resize(bRows);

  const std::size_t biasesBytes = sizeof(Scalar) * bRows;

  ifs.read((char*) biases_.data(), biasesBytes);

  int meanRows;
  ifs.read((char*) (&meanRows), sizeof(int));

  runningMean_.resize(meanRows);
  runningVariance_.resize(meanRows);

  const std::size_t meanBytes = sizeof(Scalar) * meanRows;

  ifs.read((char*) runningMean_.data(), meanBytes);
  ifs.read((char*) runningVariance_.data(), meanBytes);
}

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
