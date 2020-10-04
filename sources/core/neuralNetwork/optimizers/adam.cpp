#include <iostream>
#include <math.h>

#include "primitives/Array.h"

#include "adam.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


adam::adam(Scalar alpha, Scalar decayM1, Scalar decayM2):
  learningRate_(alpha),
  decayM1_(decayM1),
  decayM2_(decayM2),
  decayM1t_(decayM1),
  decayM2t_(decayM2),
  epsilon_(1e-12)
{
  if (learningRate_ <= 0)
  {
    std::cerr << "Learning rate must be > 0.\n";

    std::abort();
  }

  if (decayM1_  <= 0 || decayM1_  >= 1 ||
      decayM2_  <= 0 || decayM2_  >= 1 ||
      decayM1t_ <= 0 || decayM1t_ >= 1 ||
      decayM2t_ <= 0 || decayM2t_ >= 1
    )
  {
    std::cerr << "Decay parameterrate must be > 0 and < 1.\n";

    std::abort();
  }
}

adam::adam(
           Scalar alpha,
           Scalar decayM1,
           Scalar decayM2,
           Scalar epsilon
          ):
  learningRate_(alpha),
  decayM1_(decayM1),
  decayM2_(decayM2),
  decayM1t_(decayM1),
  decayM2t_(decayM2),
  epsilon_(epsilon)
{
  if (learningRate_ <= 0)
  {
    std::cerr << "Learning rate must be > 0.\n";

    std::abort();
  }

  if (decayM1_  <= 0 || decayM1_  >= 1 ||
      decayM2_  <= 0 || decayM2_  >= 1 ||
      decayM1t_ <= 0 || decayM1t_ >= 1 ||
      decayM2t_ <= 0 || decayM2t_ >= 1
    )
  {
    std::cerr << "Decay parameterrate must be > 0 and < 1.\n";

    std::abort();
  }

  if (epsilon_ <= 0)
  {
    std::cerr << "Epsilon must be > 0.\n";

    std::abort();
  }
}

void adam::init(
                const Matrix& weightsGradient,
                const Vector& biasesGradient
               ) const
{
  Matrix& weightsMometum1
      = weightsM1History_[weightsGradient.data()];

  weightsMometum1.setZero(
                           weightsGradient.rows(),
                           weightsGradient.cols()
                          );

  Matrix& weightsMometum2
      = weightsM2History_[weightsGradient.data()];

  weightsMometum2.setZero(
                           weightsGradient.rows(),
                           weightsGradient.cols()
                          );

  if (biasesGradient.size() > 0)
  {
    Vector& biasesMomentum1
        = biasesM1History_[biasesGradient.data()];

    biasesMomentum1.setZero(biasesGradient.rows());

    Vector& biasesMomentum2
        = biasesM2History_[biasesGradient.data()];

    biasesMomentum2.setZero(biasesGradient.rows());
  }
}

void adam::update(
                  const Matrix& weightsGradient,
                  const Vector& biasesGradient,
                  Matrix& deltaWeights,
                  Vector& deltaBiases
                 ) const
{
  Matrix& weightsMomentum1
      = weightsM1History_[weightsGradient.data()];

  Matrix& weightsMomentum2
      = weightsM2History_[weightsGradient.data()];

  computeUpdate(
                weightsGradient,
                weightsMomentum1,
                weightsMomentum2,
                deltaWeights
               );

  if (biasesGradient.size() > 0)
  {
    Vector& biasesMomentum1
        = biasesM1History_[biasesGradient.data()];

    Vector& biasesMomentum2
        = biasesM2History_[biasesGradient.data()];

    computeUpdate(
                  biasesGradient,
                  biasesMomentum1,
                  biasesMomentum2,
                  deltaBiases
                 );
  }

  decayM1t_ *= decayM1_;
  decayM2t_ *= decayM2_;
}

template <class T>
void adam::computeUpdate(
                         const T& gradient,
                         T& historyM1,
                         T& historyM2,
                         T& increment
                        ) const
{
  historyM1 =  (1.0 - decayM1_) * gradient
            +         decayM1_  * historyM1;

  historyM2.array() =
      (1.0 - decayM2_) * gradient.array().square()
    +        decayM2_  * historyM2.array();

  // Corrections
  const Scalar corr1 = 1.0 / (1.0 - decayM1t_);
  const Scalar corr2 = 1.0 / (1.0 - decayM2t_);

  increment.array() = -learningRate_
                    *  corr1 * historyM1.array()
                    * (
                       corr2 * historyM2.array() + epsilon_
                      ).rsqrt();
}


} // namespace
