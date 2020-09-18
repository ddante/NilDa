#include <iostream>

#include "primitives/Array.h"

#include "rsmProp.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


rsmProp::rsmProp(Scalar alpha, Scalar decay):
  learningRate_(alpha),
  decay_(decay),
  initAccumlation_(0),
  epsilon_(1e-12)
{
  if (learningRate_ <= 0)
  {
    std::cerr << "Learning rate must be > 0.\n";

    std::abort();
  }

  if (decay_ <= 0 || decay >= 1)
  {
    std::cerr << "Decay parameterrate must be > 0 and < 1.\n";

    std::abort();
  }
}

rsmProp::rsmProp(
                 Scalar alpha,
                 Scalar decay,
                 Scalar initAccumlation
                ):
  learningRate_(alpha),
  decay_(decay),
  initAccumlation_(initAccumlation),
  epsilon_(1e-12)
{
  if (learningRate_ <= 0)
  {
    std::cerr << "Learning rate must be > 0.\n";

    std::abort();
  }

  if (decay_ <= 0 || decay_ >= 1)
  {
    std::cerr << "Decay parameterrate must be > 0 and < 1.\n";

    std::abort();
  }

  if (initAccumlation_ < 0)
  {
    std::cerr << "Initial accumulation must be >= 0.\n";

    std::abort();
  }
}

rsmProp::rsmProp(
                 Scalar alpha,
                 Scalar decay,
                 Scalar initAccumlation,
                 Scalar epsilon
                ):
  learningRate_(alpha),
  decay_(decay),
  initAccumlation_(initAccumlation),
  epsilon_(epsilon)
{
  if (learningRate_ <= 0)
  {
    std::cerr << "Learning rate must be > 0.\n";

    std::abort();
  }

  if (decay_ <= 0 || decay_ >= 1)
  {
    std::cerr << "Decay parameterrate must be > 0 and < 1.\n";

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

void rsmProp::init(
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

void rsmProp::update(
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
  weightsAccumulator.array() =
             decay_  * weightsAccumulator.array()
    + (1.0 - decay_) * weightsGradient.array().square();

  biasesAccumulator.array() =
             decay_  * biasesAccumulator.array()
    + (1.0 - decay_) * biasesGradient.array().square();

  // Return the increment of the weights and biases
  deltaWeights.array() = -learningRate_
                       * weightsGradient.array()
                       * (
                          weightsAccumulator.array()
                           + epsilon_
                         ).rsqrt();


  deltaBiases.array() = -learningRate_
                      * biasesGradient.array()
                      * (
                         biasesAccumulator.array()
                          + epsilon_
                        ).rsqrt();
}


} // namespace
