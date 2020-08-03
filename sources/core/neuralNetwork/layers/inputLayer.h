#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include <iostream>
#include <array>

#include "primitives/Vector.h"
#include "primitives/Matrix.h"
#include "primitives/errors.h"

#include "layer.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

class inputLayer : public layer
{

private:

  // Column stride for each observation.
  // It is 1 for flatten layer.
  // It is the number of channels for 2D input
  int observationStride_;

public:

  // Constructors

  inputLayer(const int inSize);

  inputLayer(const std::array<int,3>& inSize);

  // Destructor

  ~inputLayer()  = default;

  // Member functions
  void init(const layer* previousLayer) override
  {
    std::cerr << "Input layer cannot call the init." << std::endl;

    assert(false);
  }

  void checkInputSize(const Matrix& obs) const override;

  void forwardPropagation(const Matrix& obs) override
  {
    std::cerr << "Input layer cannot call forwardPropagation." << std::endl;

    assert(false);
  }

  void backwardPropagation(
                           const Matrix& dActivationNext,
                           const Matrix& inputData
                          ) override
  {
    std::cerr << "Input layer cannot call backwardPropagation." << std::endl;

    assert(false);
  }

  const Matrix& getWeights() const override
  {
    std::cerr << "Input layer cannot call getWeights." << std::endl;

    assert(false);
  }

  const Vector& getBiases() const override
  {
    std::cerr << "Input layer cannot call getBiases." << std::endl;

    assert(false);
  }

  const Matrix& getWeightsDerivative() const override
  {
    std::cerr << "Input layer cannot call getWeightsDerivative." << std::endl;

    assert(false);
  }

  const Vector& getBiasesDerivative() const override
  {
    std::cerr << "Input layer cannot call getBiasesDerivative." << std::endl;

    assert(false);
  }

  const Matrix& output() const override
  {
    std::cerr << "Input layer cannot call output." << std::endl;

    assert(false);
  }

  const Matrix& backPropCache() const override
  {
    std::cerr << "Input layer cannot call backPropCache." << std::endl;

    assert(false);
  }

  void setWeightsAndBiases(
                           const Matrix& W,
                           const Vector& b
                          ) override
  {
    std::cerr << "Input layer cannot call setWeightsAndBiases." << std::endl;

    assert(false);
  }

  void incrementWeightsAndBiases(
                                 const Matrix& deltaW,
                                 const Vector& deltaB
                                ) override
  {
    std::cerr << "Input layer cannot call incrementWeightsAndBiases." << std::endl;

    assert(false);
  }

  int inputStride() const override
  {
    return observationStride_;
  }

  errorCheck localChecks(
                         const Matrix& input,
                         Scalar errTol
                        ) const override
  {

    std::cerr << "No localChecks for input layer" << std::endl;

    assert(false);

    errorCheck output;
    return output;
  }

};


} // namespace

#endif
