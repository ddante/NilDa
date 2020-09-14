#include <iostream>

#include "primitives/Array.h"

#include "adaGrad.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


adaGrad::adaGrad(Scalar alpha):
  learningRate_(alpha),
  initAccumlation_(0),
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

  weightsAccumulator.setZero(
                             weightsGradient.rows(),
                             weightsGradient.cols()
                            );

  Vector& biasesAccumulator
      = biasesHistory_[biasesGradient.data()];

  biasesAccumulator.setZero(biasesGradient.rows());

  if (initAccumlation_ > 0)
  {
    weightsAccumulator.array() += initAccumlation_;

    biasesAccumulator.array() += initAccumlation_;
  }
}

void adaGrad::update(const Matrix& weightsGradient,
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

  // Correction of the learning rates
  Matrix corrW = weightsGradient.array()
               * (weightsAccumulator.array() + epsilon_).rsqrt();

  Matrix corrB = biasesGradient.array()
               * (biasesAccumulator.array() + epsilon_).rsqrt();

  // Return the increment of the weights and biases

  deltaWeights.noalias() = -learningRate_ * corrW;

  deltaBiases.noalias() = -learningRate_ * corrB;
}


} // namespace
