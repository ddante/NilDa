#ifndef BATCH_NORMALIZATION_LAYER_H
#define BATCH_NORMALIZATION_LAYER_H

#include <iostream>
#include <memory>

#include "primitives/Vector.h"
#include "primitives/Matrix.h"
#include "primitives/errors.h"

#include "activationFunctions/activationFunction.h"

#include "layer.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


class batchNormalizationLayer : public layer
{

private:

  // Pointer to the activation function used in this layer
  std::unique_ptr<activationFunction> activationFunction_;

  // Linear output and activation
  Matrix logit_;
  Matrix activation_;

  // Batch mean and standard deviation
  Vector batchMean_;
  Vector batchVariance_;

  // Mean of the dataset
  Vector dataSetMean_;
  Vector dataSetMean2_; // mean of the square

  // Weight and derivative w.r.t. weights
  Matrix weights_;
  Matrix dWeights_;

  // Bias and derivative w.r.t. bias vectors
  Vector biases_;
  Vector dBiases_;

  // Cache matrix to store the data to pass to the layer
  // of the previous level during the back propagation
  Matrix cacheBackProp_;

  // Store the number of observations seen in the
  // forward propagation
  int nObservations_;

  // Counter to compute the comulative moving average
  int counterCMA_;

  // Small constant used for the logit normalization
  // to avoid division by zero
  const Scalar epsilonTol_ = 1e-12;

private:

  void checkInputAndCacheSize(
                              const Matrix& inputData,
                              const Matrix& cacheBackProp
                             ) const;

  void setActivationFunction(const activationFunctions code);

public:

  // Constructor

  batchNormalizationLayer();

  // Destructor

  ~batchNormalizationLayer()  = default;

  // Member functions

  void checkInput() const override
  {
    // Nothing to do here
  }

  void setupForward(const layer* previousLayer) override;

  void setupBackward(const layer* nextLayer) override
  {
    // Nothing to be done here
  }

  void init(const bool resetWeightBiases) override;

  void checkInputSize(const Matrix& inputData) const override;

  void forwardPropagation(
                          const Matrix& inputData,
                          const bool trainingPhase
                         ) override;

  void backwardPropagation(const Matrix& dActivationNex,
                           const Matrix& inputData) override;

  const Matrix& getWeights() const override
  {
    return weights_;
  }

  const Vector& getBiases() const override
  {
    return biases_;
  }

  const Matrix& getWeightsDerivative() const override
  {
    return dWeights_;
  }

  const Vector& getBiasesDerivative() const override
  {
    return dBiases_;
  }

  void setWeightsAndBiases(
                           const Matrix& W,
                           const Vector& b
                          ) override;

  void incrementWeightsAndBiases(
                                 const Matrix& deltaW,
                                 const Vector& deltaB
                                ) override;

  const Matrix& output() const override
  {
    return activation_;
  }

  const Matrix& backPropCache() const override
  {
    return cacheBackProp_;
  }

  int inputStride() const override
  {
    std::cerr << "Batch normalization layer "
                 "cannot call inputStride function.\n";

    assert(false);
  }

  int numberOfParameters() const override
  {
    return numberOfWeights() + numberOfBiases();
  }

  int numberOfWeights() const override
  {
    return weights_.size();
  }

  int numberOfBiases() const override
  {
    return biases_.size();
  }

  void saveLayer(std::ofstream& ofs) const override;

  void loadLayer(std::ifstream& ifs) override;

  errorCheck localChecks(
                         const Matrix& input,
                         Scalar errTol
                        ) const override
  {
    std::cerr << "No localChecks for batch normalization layer.\n";

    assert(false);

    errorCheck output;
    return output;
  }
};


} // namespace

#endif
