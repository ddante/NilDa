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

  inputLayer();

  explicit inputLayer(const int inSize);

  explicit inputLayer(const std::array<int,3>& inSize);

  // Destructor

  ~inputLayer()  = default;

  // Member functions

  void init(
            const layer* previousLayerconst,
            const bool resetWeightBiases
           ) override
  {
    std::cerr << "Input layer cannot call the init.\n";

    assert(false);
  }

  void checkInput() const override;

  void setupBackward(const layer* nextLayer) override
  {
    std::cerr << "Input layer cannot call setupBackward.\n";

    assert(false);
  }

  void checkInputSize(const Matrix& obs) const override;

  void forwardPropagation(const Matrix& obs) override
  {
    std::cerr << "Input layer cannot call forwardPropagation.\n";

    assert(false);
  }

  void backwardPropagation(
                           const Matrix& dActivationNext,
                           const Matrix& inputData
                          ) override
  {
    std::cerr << "Input layer cannot call backwardPropagation.\n";

    assert(false);
  }

  const Matrix& getWeights() const override
  {
    std::cerr << "Input layer cannot call getWeights.\n";

    assert(false);
  }

  const Vector& getBiases() const override
  {
    std::cerr << "Input layer cannot call getBiases.\n";

    assert(false);
  }

  const Matrix& getWeightsDerivative() const override
  {
    std::cerr << "Input layer cannot call getWeightsDerivative.\n";

    assert(false);
  }

  const Vector& getBiasesDerivative() const override
  {
    std::cerr << "Input layer cannot call getBiasesDerivative.\n";

    assert(false);
  }

  const Matrix& output() const override
  {
    std::cerr << "Input layer cannot call output.\n";

    assert(false);
  }

  const Matrix& backPropCache() const override
  {
    std::cerr << "Input layer cannot call backPropCache.\n";

    assert(false);
  }

  void setWeightsAndBiases(
                           const Matrix& W,
                           const Vector& b
                          ) override
  {
    std::cerr << "Input layer cannot call setWeightsAndBiases.\n";

    assert(false);
  }

  void incrementWeightsAndBiases(
                                 const Matrix& deltaW,
                                 const Vector& deltaB
                                ) override
  {
    std::cerr << "Input layer cannot call incrementWeightsAndBiases.\n";

    assert(false);
  }

  int inputStride() const override
  {
    return observationStride_;
  }

  int numberOfParameters() const override
  {
    return 0;
  }

  void saveLayer(std::ofstream& ofs) const override;

  void loadLayer(std::ifstream& ifs) override;

  errorCheck localChecks(
                         const Matrix& input,
                         Scalar errTol
                        ) const override
  {

    std::cerr << "No localChecks for input layer\n";

    assert(false);

    errorCheck output;
    return output;
  }

};


} // namespace

#endif
