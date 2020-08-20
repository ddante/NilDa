#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

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

class denseLayer : public layer
{

private:

  // Number of neurons of the current layer
  int layerSize_;

  // Pointer to the activation function used in this layer
  std::unique_ptr<activationFunction> activationFunction_;

  // Linear output and activation matrices
  Matrix linearOutput_;
  Matrix activation_;

  // Weight and derivative w.r.t. weights matrices
  Matrix weights_;
  Matrix dWeights_;

  // Bias and derivative w.r.t. bias vectors
  Vector biases_;
  Vector dBiases_;

  // Cache matrix to store the data to pass to the layer
  // of the previous level during the back propagation
  Matrix cacheBackProp_;

  // Store if the flattening is needed when
  // the input comes from a not flat layer
  bool needFlattening_;

  // Store the size and the number of channels of
  // the previous not flat layers. Neded to flatten the input.
  int inputSize_;
  int inputChannels_;

  // Store the number of observations seen in the
  // forward propagation
  int nObservations_;

public:

  // Constructor

  denseLayer(
             const int inSize,
             const std::string& activationName
            );

  // Destructor

  ~denseLayer()  = default;

  // Member functions

  void init(const layer* previousLayer) override;

  void setupBackward(const layer* nextLayer) override
  {
    // Nothing to be done here
  }

  void checkInputSize(const Matrix& inputData) const override;

  void checkInputAndCacheSize(const Matrix& inputData,
                              const Matrix& cacheBackProp) const;

  void forwardPropagation(const Matrix& inputData)  override;

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
    std::cerr << "Dense layer cannot call inputStride function" << std::endl;

    assert(false);
  }

  int numberOfParameters() const override
  {
    return weights_.size();
  }

  errorCheck localChecks(
                         const Matrix& input,
                         Scalar errTol
                        ) const override
  {
    std::cerr << "No localChecks for dense layer" << std::endl;

    assert(false);

    errorCheck output;
    return output;
  }

};


} // namespace

#endif
