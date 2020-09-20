#include <iostream>

#include "primitives/Array.h"

#include "adaGrad.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


adaGrad::adaGrad(Scalar alpha):
  learningRate_(alpha),
  initAccumlation_(0.1),
  epsilon_(1e-12)
{
  if (learningRate_ <= 0)
  {
    std::cerr << "Learning rate must be > 0.\n";

    std::abort();
  }
}

adaGrad::adaGrad(Scalar alpha, Scalar initAccumlation):
  learningRate_(alpha),
  initAccumlation_(initAccumlation),
  epsilon_(1e-12)
{
  if (learningRate_ <= 0)
  {
    std::cerr << "Learning rate must be > 0.\n";

    std::abort();
  }

  if (initAccumlation_ < 0)
  {
    std::cerr << "Initial accumulation must be >= 0.\n";

    std::abort();
  }
}

adaGrad::adaGrad(Scalar alpha, Scalar initAccumlation, Scalar epsilon):
  learningRate_(alpha),
  initAccumlation_(initAccumlation),
  epsilon_(epsilon)
{
  if (learningRate_ <= 0)
  {
    std::cerr << "Learning rate must be > 0.\n";

    std::abort();
  }

  if (initAccumlation_ < 0)
  {
    std::cerr << "Initial accumulation must be >= 0.\n";

    std::abort();
  }

  if (epsilon_ <= 0)
  {
    std::cerr << "Epsilon must be > 0.\n";

    std::abort();
  }
}

void adaGrad::init(
                   const Matrix& weightsGradient,
                   const Vector& biasesGradient
                  ) const
{
  Matrix& weightsAccumulator
      = weightsHistory_[weightsGradient.data()];

  weightsAccumulator.setConstant(
                                 weightsGradient.rows(),
                                 weightsGradient.cols(),
                                 initAccumlation_
                                );

  Vector& biasesAccumulator
      = biasesHistory_[biasesGradient.data()];

  biasesAccumulator.setConstant(
                                biasesGradient.rows(),
                                initAccumlation_
                               );
}

void adaGrad::update(
                     const Matrix& weightsGradient,
                     const Vector& biasesGradient,
                     Matrix& deltaWeights,
                     Vector& deltaBiases
                    ) const
{
  // Get the history of the weights and biases
  // associated with the current layer
  Matrix& weightsAccumulator
      = weightsHistory_[weightsGradient.data()];

  Vector& biasesAccumulator
      = biasesHistory_[biasesGradient.data()];

  // Update the weights and biases using momentum
  weightsAccumulator.array() +=  weightsGradient.array().square();

  biasesAccumulator.array() += biasesGradient.array().square();

  // Return the increment of the weights and biases

  deltaWeights.array() = -learningRate_
                       *  weightsGradient.array()
                       * (
                          weightsAccumulator.array()
                          + epsilon_
                         ).rsqrt();


  deltaBiases.array() = -learningRate_
                      *  biasesGradient.array()
                      * (
                         biasesAccumulator.array()
                         + epsilon_
                        ).rsqrt();
}


} // namespace
