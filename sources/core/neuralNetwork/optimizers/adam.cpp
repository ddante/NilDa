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

  Vector& biasesMomentum1
      = biasesM1History_[biasesGradient.data()];

  biasesMomentum1.setZero(biasesGradient.rows());

  Vector& biasesMomentum2
      = biasesM2History_[biasesGradient.data()];

  biasesMomentum2.setZero(biasesGradient.rows());
}

void adam::update(
                  const Matrix& weightsGradient,
                  const Vector& biasesGradient,
                  Matrix& deltaWeights,
                  Vector& deltaBiases
                 ) const
{
  // Get the history of the weights and biases
  // associated with the current layer
  Matrix& weightsMomentum1
      = weightsM1History_[weightsGradient.data()];

  Matrix& weightsMomentum2
      = weightsM2History_[weightsGradient.data()];

  Vector& biasesMomentum1
      = biasesM1History_[biasesGradient.data()];

  Vector& biasesMomentum2
      = biasesM2History_[biasesGradient.data()];

  // Update the weights and biases using momentum
  weightsMomentum1 =        decayM1_  * weightsMomentum1
                   + (1.0 - decayM1_) * weightsGradient;

  biasesMomentum1 =        decayM1_  * biasesMomentum1
                  + (1.0 - decayM1_) * biasesGradient;

  weightsMomentum2.array() =
                 decayM2_  * weightsMomentum2.array()
        + (1.0 - decayM2_) * weightsGradient.array().square();

  biasesMomentum2.array() =
                decayM2_  * biasesMomentum2.array()
       + (1.0 - decayM2_) * biasesGradient.array().square();

  // Corrections

  const Scalar corr1 = 1.0 / (1.0 - decayM1t_);
  const Scalar corr2 = 1.0 / (1.0 - decayM2t_);

  // Return the increment of the weights and biases
  deltaWeights.array() = -learningRate_
                       *  corr1 * weightsMomentum1.array()
                       * (
                          corr2 * weightsMomentum2.array()
                           + epsilon_
                         ).rsqrt();

  deltaBiases.array() = -learningRate_
                      *  corr1 * biasesMomentum1.array()
                      * (
                         corr2 * biasesMomentum2.array()
                          + epsilon_
                        ).rsqrt();

  decayM1t_ *= decayM1_;
  decayM2t_ *= decayM2_;
}


} // namespace
