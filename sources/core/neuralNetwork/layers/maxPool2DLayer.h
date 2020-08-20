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

  // Specify if, in the back propagation, the output
  // the input from next layer must be reshaped.
  // This is necessary if the conv2D layer is connected
  // to a dense layer
  bool undoFlattening_;

  // Store the number of observations seen in the
  // forward propagation
  //int nObservations_;

  // Store all the quanities required for the pooling
  pool2DDimensions poolDims_;

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

  void setupBackward(const layer* nextLayer) override;

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
                        ) const override;
};


} // namespace

#endif
