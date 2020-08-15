#ifndef MAX_POOL_2D_H
#define MAX_POOL_2D_H

#include <iostream>
#include <array>

#include "pool2DUtils.h"

#include "layer.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


class maxPool2DLayer : public layer
{

private:

  // Dimensions of the kernel (rows, cols)
  std::array<int, 2> kernelSize_;

  // Stride of the kernel (rows, cols)
  std::array<int, 2> kernelStride_;

  // Output of the polling and activation
  Matrix linearOutput_;

  // Stroe the indices of the max values
  MatrixI maxIndices_;

  // cache for the back propagation
  Matrix cacheBackProp_;

  pool2DDimensions poolDims_;

  // Store the number of observations seen in the
  // forward propagation
  int nObservations_;

public:

  // Constructor

  maxPool2DLayer(
                 const std::array<int, 2>& poolSize,
    			       const std::array<int, 2>& poolStride
                );

  // Destructor

  ~maxPool2DLayer() = default;

  // Member functions

  void init(const layer* previousLayer) override;

  void setupBackward(const layer* nextLayer) override
  {
    // Nothing to be done here.
  }

  void checkInputSize(const Matrix& inputData) const override;

  void checkInputAndCacheSize(
                              const Matrix& input,
                              const Matrix& cacheBackProp
                             ) const;

  void forwardPropagation(const Matrix& input) override;

  void backwardPropagation(const Matrix& dActivationNext,
                           const Matrix& input) override;

  const Matrix& getWeights() const override
  {
    std::cerr << "Max pool 2D layer cannot call "
              << "getWeights function" << std::endl;

    assert(false);
  }

  const Vector& getBiases() const override
  {
    std::cerr << "Max pool 2D layer cannot call "
              << "getBiases function" << std::endl;

    assert(false);
  }

  const Matrix& getWeightsDerivative() const override
  {
    std::cerr << "Max pool 2D layer cannot call "
              << "getWeightsDerivative function" << std::endl;

    assert(false);
  }

  const Vector& getBiasesDerivative() const override
  {
    std::cerr << "Max pool 2D layer cannot call "
              << "getBiasesDerivative function" << std::endl;

    assert(false);
  }

  void setWeightsAndBiases(
                           const Matrix& W,
                           const Vector& b
                         ) override
  {
    std::cerr << "Max pool 2D layer cannot call "
              << "setWeightsAndBiases function" << std::endl;

    assert(false);
  }

  void incrementWeightsAndBiases(
                                 const Matrix& deltaW,
                                 const Vector& deltaB
                                ) override
  {
    std::cerr << "Max pool 2D layer cannot call "
              << "incrementWeightsAndBiases function" << std::endl;

    assert(false);
  }

  const Matrix& output() const override
  {
    return linearOutput_;
  }

  const Matrix& backPropCache() const override
  {
    return cacheBackProp_;
  }

  int inputStride() const override
  {
    std::cerr << "Max pool 2D layer cannot call "
              << "inputStride function" << std::endl;

    assert(false);
  }

  int numberOfParameters() const override
  {
    return 0;
  }

  errorCheck localChecks(
                         const Matrix& input,
                         const Scalar errorLimit
                        ) const override
  {
    std::cerr << "No localChecks for dMax pool 2D layer" << std::endl;

    assert(false);

    errorCheck output;
    return output;
  }
};


} // namespace

#endif
