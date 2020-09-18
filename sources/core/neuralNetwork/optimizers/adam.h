#ifndef ADAM_H
#define ADAM_H

#include <iostream>
#include <memory>
#include <map>

#include "optimizer.h"

#include "primitives/Scalar.h"
#include "primitives/Array.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

class adam : public optimizer
{
private:

  Scalar learningRate_;

  // Discounting factor for the history of the gradient
  Scalar decayM1_, decayM2_;

  // Corrected discounting factor for the history of the gradient
  mutable Scalar decayM1t_, decayM2t_;

  // A small value to avoid zero denominator
  Scalar epsilon_;

  // To recover the correct history of the weights and
  // biases associated with their respective gradient,
  // store the history in map using as key a constant pointer
  // to the gradients of the weights and biases
  mutable std::map<const Scalar*, Matrix> weightsM1History_;

  mutable std::map<const Scalar*, Matrix> weightsM2History_;

  mutable std::map<const Scalar*, Vector> biasesM1History_;

  mutable std::map<const Scalar*, Vector> biasesM2History_;

public:

  // Constructors

  adam() = delete;

  adam(Scalar alpha, Scalar decayM1, Scalar decayM2);

  adam(Scalar alpha, Scalar decayM1, Scalar decayM2, Scalar epsilon);

  // Member functions

  void get() const override
  {
    std::cout << "Learning rate: " << learningRate_ << ", "
              << "Decay 1st momentum: " << decayM1_ << ", "
              << "Decay 2nd momentum: " << decayM2_ << ", "
              << "Epsilon: " << epsilon_ << "\n";
  }

  void init(
            const Matrix& weightsGradient,
            const Vector& biasesGradient
           ) const override;


  void update(const Matrix& weightsGradient,
              const Vector& biasesGradient,
              Matrix& deltaWeights,
              Vector& deltaBiases
             ) const override;

  // Destructor

  ~adam() = default;
};


}// namespace

#endif
