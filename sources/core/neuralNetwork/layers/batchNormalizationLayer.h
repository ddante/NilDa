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

  // Linear output and activation matrices
  Matrix logit_;
  Matrix activation_;

  // Weight and derivative w.r.t. weights matrices
  Matrix weights_;
  Matrix dWeights_;

  // Cache matrix to store the data to pass to the layer
  // of the previous level during the back propagation
  Matrix cacheBackProp_;

private:

  void setActivationFunction(const activationFunctions code);

public:

  // Constructor

  batchNormalizationLayer();

  // Destructor

  ~batchNormalizationLayer()  = default;

  // Member functions

  void checkInput() const
  {
    // Nothing to do here
  }

  void init(
            const layer* previousLayer,
            const bool resetWeightBiases
           ) override;

  void setupBackward(const layer* nextLayer) override
  {
    // Nothing to be done here
  }

  void checkInputSize(const Matrix& inputData) const override
  {
    // Nothing to be done here
  }

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
    std::cerr << "Batch normalization layer "
              << "cannot call getBiases function.\n";

    assert(false);
  }

  const Matrix& getWeightsDerivative() const override
  {
    return dWeights_;
  }

  const Vector& getBiasesDerivative() const override
  {
    std::cerr << "Batch normalization layer "
              << "cannot call getBiasesDerivative function.\n";

    assert(false);
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
    return weights_.size();
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
