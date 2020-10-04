#include <iostream>

#include "primitives/Array.h"

#include "rmsProp.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


rmsProp::rmsProp(Scalar alpha, Scalar decay):
  learningRate_(alpha),
  decay_(decay),
  initAccumlation_(0.1),
  epsilon_(1e-8)
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

rmsProp::rmsProp(
                 Scalar alpha,
                 Scalar decay,
                 Scalar initAccumlation
                ):
  learningRate_(alpha),
  decay_(decay),
  initAccumlation_(initAccumlation),
  epsilon_(1e-8)
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

rmsProp::rmsProp(
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

void rmsProp::init(
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

  if (biasesGradient.size() > 0)
  {
    Vector& biasesAccumulator
        = biasesHistory_[biasesGradient.data()];

    biasesAccumulator.setConstant(
                                  biasesGradient.rows(),
                                  initAccumlation_
                                 );
  }
}

void rmsProp::update(
                     const Matrix& weightsGradient,
                     const Vector& biasesGradient,
                     Matrix& deltaWeights,
                     Vector& deltaBiases
                    ) const
{
  Matrix& weightsAccumulator
      = weightsHistory_[weightsGradient.data()];

  computeUpdate(
                weightsGradient,
                weightsAccumulator,
                deltaWeights
               );

  if (biasesGradient.size() > 0)
  {
    Vector& biasesAccumulator
        = biasesHistory_[biasesGradient.data()];

    computeUpdate(
                  biasesGradient,
                  biasesAccumulator,
                  deltaBiases
                 );
  }
}

template <class T>
void rmsProp::computeUpdate(
                            const T& gradient,
                            T& accumulator,
                            T& increment
                           ) const
{
  // Exponential weighted sqaure gradient
  accumulator.array() =
      (1.0 - decay_) * gradient.array().square()
           + decay_  * accumulator.array();

  // Increment using the scaled gradient
  increment.array() = -learningRate_
                    *  gradient.array()
                    * (
                       accumulator.array() + epsilon_
                      ).rsqrt();
}


} // namespace
