#include <iostream>

#include "primitives/Array.h"

#include "sgd.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

sgd::sgd(Scalar alpha):
  learningRate_(alpha),
  momentum_(0.0)
{
  if (learningRate_ <= 0)
  {
    std::cerr << "Learning rate must be > 0.\n";

    std::abort();
  }
}

sgd::sgd(Scalar alpha, Scalar m):
  learningRate_(alpha),
  momentum_(m)
{
  if (learningRate_ <= 0)
  {
    std::cerr << "In SGD optimizer, "
              << "learning rate must be > 0.\n";

    std::abort();
  }

  if (momentum_ < 0 || momentum_ >=1)
  {
    std::cerr << "In SGD optimizer, "
                 "momentum must be >= 0 and < 1 " << std::endl;

    std::abort();
  }
}

void sgd::init(
               const Matrix& weightsGradient,
               const Vector& biasesGradient
              ) const
{
  Matrix& weightsFirstMomentum
      = weightsHistory_[weightsGradient.data()];

  weightsFirstMomentum.setZero(
                               weightsGradient.rows(),
                               weightsGradient.cols()
                              );

  if (biasesGradient.size() > 0)
  {
    Vector& biasesFirstMomentum
        = biasesHistory_[biasesGradient.data()];

    biasesFirstMomentum.setZero(biasesGradient.rows());
  }
}

void sgd::update(
                 const Matrix& weightsGradient,
                 const Vector& biasesGradient,
                 Matrix& deltaWeights,
                 Vector& deltaBiases
                ) const
{
  Matrix& weightsFirstMomentum
      = weightsHistory_[weightsGradient.data()];

  computeUpdate(
                weightsGradient,
                weightsFirstMomentum,
                deltaWeights
               );

  if (biasesGradient.size() > 0)
  {
    Vector& biasesFirstMomentum
        = biasesHistory_[biasesGradient.data()];

    computeUpdate(
                  biasesGradient,
                  biasesFirstMomentum,
                  deltaBiases
                 );
  }
}

template <class T>
void sgd::computeUpdate(
                        const T& gradient,
                        T& firstMomentum,
                        T& increment
                       ) const
{
  // Update using momentum
  firstMomentum = (1.0 - momentum_) * gradient
                +        momentum_  * firstMomentum;

  // Return the increment
  increment = -learningRate_ * firstMomentum;
}


} // namespace
