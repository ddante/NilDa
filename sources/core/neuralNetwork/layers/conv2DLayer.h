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

  conv2DDimensions forwardConvDims_;
  conv2DDimensions backwardWeightsConvDims_;
  conv2DDimensions backwardInputConvDims_;

  // Store the number of observations seen in the
  // forward propagation
  int nObservations_;

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

  void checkInputSize(const Matrix& inputData) const override;

  void forwardPropagation(const Matrix& input) override;

  void backwardPropagation(const Matrix& dActivationNex,
                           const Matrix& inputData) override;

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

  int size() const override
  {
    return  forwardConvDims_.outputRows
          * forwardConvDims_.outputCols
          * forwardConvDims_.outputChannels;
  }

  void size(std::array<int, 3>& sizes) const override
  {
      sizes =
      {
        forwardConvDims_.outputRows,
        forwardConvDims_.outputCols,
        forwardConvDims_.outputChannels
      };
  }

  int inputStride() const override
  {
    std::cerr << "Conv 2d layer cannot call inputStride function" << std::endl;

    assert(false);
  }

  errorCheck localChecks(
                         const Matrix& input,
                         const Scalar errorLimit
                        ) const override;
};


} // namespace

#endif
