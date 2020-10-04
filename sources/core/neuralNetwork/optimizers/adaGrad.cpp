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

adaGrad::adaGrad(
                 Scalar alpha,
                 Scalar initAccumlation,
                 Scalar epsilon
                ):
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

void adaGrad::update(
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
void adaGrad::computeUpdate(
                            const T& gradient,
                            T& accumulator,
                            T& increment
                          ) const
{
  // Accumulate the squared grdient
  accumulator.array() += gradient.array().square();

  // Increment using the scaled gradient
  increment.array() = -learningRate_
                    *  gradient.array()
                    * (
                       accumulator.array() + epsilon_
                      ).rsqrt();
}


} // namespace
