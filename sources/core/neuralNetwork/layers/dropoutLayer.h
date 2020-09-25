#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include <iostream>
#include <memory>

#include "primitives/Vector.h"
#include "primitives/Matrix.h"
#include "primitives/errors.h"

#include "layer.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


class dropoutLayer : public layer
{

private:

  // Probability of dropping neurons
  Scalar dropProbability_;

  // Mask used to apply the dropout
  Matrix mask_;

  // Linear output and activation matrices
  Matrix activation_;

  // Cache matrix to store the data to pass to the layer
  // of the previous level during the back propagation
  Matrix cacheBackProp_;

public:

  // Constructor

  dropoutLayer();

  dropoutLayer(const Scalar keepProbability);

  // Destructor

  ~dropoutLayer()  = default;

  // Member functions

  void checkInput() const override;

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
    std::cerr << "Dropout layer cannot call getWeights function.\n";

    std::abort();
  }

  const Vector& getBiases() const override
  {
    std::cerr << "Dropout layer cannot call getBiases function.\n";

    std::abort();
  }

  const Matrix& getWeightsDerivative() const override
  {
    std::cerr << "Dropout layer cannot call getWeightsDerivative function.\n";

    std::abort();
  }

  const Vector& getBiasesDerivative() const override
  {
    std::cerr << "Dropout layer cannot call getBiasesDerivative function.\n";

    std::abort();
  }

  void setWeightsAndBiases(
                           const Matrix& W,
                           const Vector& b
                          ) override
  {
    std::cerr << "Dropout layer cannot call getBiasesDerivative function.\n";

    std::abort();
  }

  void incrementWeightsAndBiases(
                                 const Matrix& deltaW,
                                 const Vector& deltaB
                                ) override
  {
    std::cerr << "Dropout layer cannot call getBiasesDerivative function.\n";

    std::abort();
  }

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
    std::cerr << "Dropout layer cannot call inputStride function.\n";

    std::abort();
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
    std::cerr << "No localChecks for dropout layer.\n";

    std::abort();

    errorCheck output;
    return output;
  }

};


} // namespace

#endif
