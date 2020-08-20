#ifndef CONV_2D_H
#define CONV_2D_H

#include <iostream>
#include <array>
#include <memory>

#include "primitives/Vector.h"
#include "primitives/Matrix.h"
#include "primitives/errors.h"

#include "activationFunctions/activationFunction.h"

#include "conv2DUtils.h"

#include "layer.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


class conv2DLayer : public layer
{

private:

  // Number of filters
  int numberOfFilters_;

  // Dimensions of the filters (rows, cols)
  std::array<int, 2> filterSize_;

  // Stride of the filters (rows, cols)
  std::array<int, 2> filterStride_;

  // Is padding used?
  bool withPadding_;

  // Name of the activation function
  std::string activationName_;

  // Pointer to the activation function used in this layer
  std::unique_ptr<activationFunction> activationFunction_;

  // Linear output and activation matrices
  Matrix linearOutput_;
  Matrix activation_;

  // Weight and derivative w.r.t. weights matrices
  Matrix filterWeights_;
  Matrix dFilterWeights_;

  // Bias and derivative w.r.t. bias vectors
  Vector biases_;
  Vector dBiases_;

  // Cache matrix to store the data to pass to the layer
  // of the previous level during the back propagation
  Matrix cacheBackProp_;

  // Specify if, in the back propagation, the output
  // the input from next layer must be reshaped.
  // This is necessary if the conv2D layer is connected
  // to a dense layer
  bool undoFlattening_;

  // Store the number of observations seen in the
  // forward propagation
  int nObservations_;

  // Store all the quanities required for conv2D in:
  //  - the forward step
  //  - computation of the weight derivative
  //  - compuation of input derivative (cache backprop)
  conv2DDimensions forwardConvDims_;
  conv2DDimensions backwardWeightsConvDims_;
  conv2DDimensions backwardInputConvDims_;

public:

  // Constructor

  conv2DLayer(
              const int numberOfFilters,
    			    const std::array<int, 2>& filterSize,
    			    const std::array<int, 2>& filterStride,
    			    const bool withPadding,
              const std::string& activationName
             );

  conv2DLayer(
              const int numberOfFilters,
    			    const std::array<int, 2>& filterSize,
    			    const std::array<int, 2>& filterStride,
              const std::string& activationName
             );

  conv2DLayer(
              const int numberOfFilters,
    			    const std::array<int, 2>& filterSize,
    			    const bool withPadding,
              const std::string& activationName
             );

  conv2DLayer(
              const int numberOfFilters,
    			    const std::array<int, 2>& filterSize,
              const std::string& activationName
             );

  // Destructor

  ~conv2DLayer() = default;

  // Member functions

  void init(const layer* previousLayer) override;

  void setupBackward(const layer* nextLayer) override;

  void checkInputSize(const Matrix& inputData) const override;

  void forwardPropagation(const Matrix& input) override;

  void backwardPropagation(const Matrix& dActivationNext,
                           const Matrix& input) override;

  const Matrix& getWeights() const override
  {
    return filterWeights_;
  }

  const Vector& getBiases() const override
  {
    return biases_;
  }

  const Matrix& getWeightsDerivative() const override
  {
    return dFilterWeights_;
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
    std::cerr << "Conv 2d layer cannot call inputStride function" << std::endl;

    assert(false);
  }

  int numberOfParameters() const override
  {
    return filterWeights_.size();
  }

  errorCheck localChecks(
                         const Matrix& input,
                         const Scalar errorLimit
                        ) const override;
};


} // namespace

#endif
