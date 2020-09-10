#include <iostream>
#include <math.h>

#include "denseLayer.h"

#include "activationFunctions/activationFunctionUtils.h"

#include "activationFunctions/identity.h"
#include "activationFunctions/sigmoid.h"
#include "activationFunctions/relu.h"
#include "activationFunctions/softmax.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

denseLayer::denseLayer():
    layerSize_(0),
    needFlattening_(false),
    inputSize_(0),
    inputChannels_(0),
    nObservations_(0)
{
  type_ = layerTypes::dense;

  size_.isFlat = true;
  size_.size = 0;
  size_.rows = 0;
  size_.cols = 0;
  size_.channels = 0;

  trainable_ = true;
}

denseLayer::denseLayer(
                       const int inSize,
                       const std::string& activationName
                      ):
    layerSize_(inSize),
    needFlattening_(false),
    inputSize_(0),
    inputChannels_(0),
    nObservations_(0)
{
  type_ = layerTypes::dense;

  size_.isFlat = true;
  size_.size = inSize;
  size_.rows = 0;
  size_.cols = 0;
  size_.channels = 0;

  trainable_ = true;

  setActivationFunction(
    activationFunctionCode(activationName)
  );

  if (layerSize_ == 1 &&
      activationFunctionCode(activationName) ==
      activationFunctions::softmax)
  {
    std::cout << "Softmax activation requires layer size > 1.\n";
    assert(false);
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

void denseLayer::init(
                      const layer* previousLayer,
                      const bool resetWeightBiases
                      )
{
  // Check that the previous layer is compatible
  // with the current layer
  if (
      previousLayer->layerType() != layerTypes::input  &&
      previousLayer->layerType() != layerTypes::dense  &&
      previousLayer->layerType() != layerTypes::conv2D &&
      previousLayer->layerType() != layerTypes::maxPool2D
     )
  {
    std::cerr << "Previous layer of type "
              <<  getLayerName(previousLayer->layerType())
              << " not compatible with current layer of type "
              << getLayerName(type_) << ".\n";

    assert(false);
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

  if (resetWeightBiases)
  {
    const Scalar epsilonInit = sqrt(6.0)
                             / sqrt(layerSize_ + inputSize_);

    weights_.setRandom(layerSize_, inputSize_);
    weights_ *= epsilonInit;

    biases_.setRandom(layerSize_);
    biases_ *= epsilonInit;
  }

  dWeights_.setZero(layerSize_, inputSize_);

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
    << weights_.cols() << ").\n";

    assert(false);
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

    assert(false);
  }
}

void denseLayer::forwardPropagation(const Matrix& inputData)
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

    linearOutput_.resize(
                          weights_.rows(),
                          input.cols()
                         );

     // Apply the weights of the layer to the input
     linearOutput_.noalias() = weights_ * input;
  }
  else
  {
#ifdef ND_DEBUG_CHECKS
    checkInputSize(inputData);
#endif

    // For a flat input layer the number of colums is nObs
    nObservations_ = inputData.cols();

    linearOutput_.resize(
                          weights_.rows(),
                          inputData.cols()
                        );

     // Apply the weights of the layer to the input
     linearOutput_.noalias() = weights_ * inputData;
  }

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
  Matrix dLinearOutput(
                       linearOutput_.rows(),
                       linearOutput_.cols()
                      );

  activationFunction_->applyBackward(
                                     linearOutput_,
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
              << weights_.cols() << ").\n";

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
            << weights_.cols() << ").\n";

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

  weights_ += deltaW;

  biases_ += deltaB;
}

void denseLayer::saveLayer(std::ofstream& ofs) const
{
  const int iType =
    static_cast<std::underlying_type_t<layerTypes> >(layerTypes::dense);

  ofs.write((char*) (&iType), sizeof(int));

  ofs.write((char*) (&trainable_), sizeof(bool));

  ofs.write((char*) (&size_.size), sizeof(int));

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
}

void denseLayer::loadLayer(std::ifstream& ifs)
{
  ifs.read((char*) (&trainable_), sizeof(bool));

  ifs.read((char*) (&size_.size), sizeof(int));

  layerSize_ = size_.size;

  size_.rows = 0;
  size_.cols = 0;
  size_.channels = 0;

  int code;
  ifs.read((char*) (&code), sizeof(int));

  activationFunctions aType =
    static_cast<activationFunctions>(code);

  setActivationFunction(aType);

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
       assert(false);
  }
}

} // namespace
