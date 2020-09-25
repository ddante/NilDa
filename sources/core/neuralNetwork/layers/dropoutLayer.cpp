#include <iostream>
#include <random>

#include "dropoutLayer.h"

#include "activationFunctions/activationFunctionUtils.h"

#include "activationFunctions/identity.h"
#include "activationFunctions/sigmoid.h"
#include "activationFunctions/relu.h"
#include "activationFunctions/softmax.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

dropoutLayer::dropoutLayer():
  dropProbability_(0)
{
  type_ = layerTypes::dropout;

  size_.isFlat = true;
  size_.size = 0;
  size_.rows = 0;
  size_.cols = 0;
  size_.channels = 0;

  trainable_ = false;
}

dropoutLayer::dropoutLayer(const Scalar dropProbability):
  dropProbability_(dropProbability)
{
  type_ = layerTypes::dropout;

  size_.isFlat = true;
  size_.size = 0;
  size_.rows = 0;
  size_.cols = 0;
  size_.channels = 0;

  trainable_ = false;
}

void dropoutLayer::checkInput() const
{
  assert(dropProbability_ > 0 && dropProbability_ < 1);
}

void dropoutLayer::init(
                        const layer* previousLayer,
                        const bool resetWeightBiases
                       )
{
  // Check that the previous layer is compatible
  // with the current layer
  if (
      previousLayer->layerType() != layerTypes::dense  &&
      previousLayer->layerType() != layerTypes::conv2D &&
      previousLayer->layerType() != layerTypes::maxPool2D
     )
  {
    std::cerr << "Previous layer of type "
              <<  getLayerName(previousLayer->layerType())
              << " not compatible with current layer of type "
              << getLayerName(type_) << ".\n";

    std::abort();
  }

  const layerSizes prevLayer = previousLayer->size();

  // Set the size of the dropout layer as the one
  // of the previous layer
  size_.isFlat = prevLayer.isFlat;
  size_.size = prevLayer.size;
  size_.rows = prevLayer.rows;
  size_.cols = prevLayer.cols;
  size_.channels = prevLayer.channels;
}

void dropoutLayer::forwardPropagation(
                                      const Matrix& inputData,
                                      const bool trainingPhase
                                     )
{
  activation_.resize(
                     inputData.rows(),
                     inputData.cols()
                    );

  // At the training phase apply droput
  if (trainingPhase)
  {
    mask_.resize(inputData.rows(), inputData.cols());

    // This has to be replace by random Matrix operator
    std::random_device rand;
    //std::mt19937 genRand(rand());
    std::mt19937 genRand(12);

    std::uniform_real_distribution<Scalar> distr(0, 1);

    mask_ = mask_.unaryExpr(
                            [&](Scalar dummy)
                            {
                              return distr(genRand);
                            }
                           );
    // ..................................................

    // Shut down some neurons
    activation_.array() =
      (mask_.array() < dropProbability_).select(0, inputData);

    // Scale the value of the active neurons
    activation_ *= 1.0/(1.0 - dropProbability_);
  }
  else
  {
    // Don't use droput at test phase
    activation_ = inputData;
  }
}

void dropoutLayer::backwardPropagation(
                                       const Matrix& dActivationNext,
                                       const Matrix& inputData
                                      )
{
  cacheBackProp_.array() =
    (mask_.array() < dropProbability_).select(0, dActivationNext);

  cacheBackProp_ *= 1.0/(1.0 - dropProbability_);
}

void dropoutLayer::saveLayer(std::ofstream& ofs) const
{
}

void dropoutLayer::loadLayer(std::ifstream& ifs)
{
}



} // namespace
