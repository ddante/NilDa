#ifndef ADA_GRAD_H
#define ADA_GRAD_H

#include <iostream>
#include <memory>
#include <map>

#include "optimizer.h"

#include "primitives/Scalar.h"
#include "primitives/Array.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

class adaGrad : public optimizer
{
private:

  Scalar learningRate_;

  // Starting value for the accumulators
  Scalar initAccumlation_;

  // A small value to avoid zero denominator
  Scalar epsilon_;

  // To recover the correct history of the weights and
  // biases associated with their respective gradient,
  // store the history in map using as key a constant pointer
  // to the gradients of the weights and biases
  mutable std::map<const Scalar*, Matrix> weightsHistory_;

  mutable std::map<const Scalar*, Vector> biasesHistory_;

public:

  // Constructors

  adaGrad() = delete;

  explicit adaGrad(Scalar alpha);

  adaGrad(Scalar alpha, Scalar initAccumlation);

  adaGrad(Scalar alpha, Scalar initAccumlation, Scalar epsilon);

  // Member functions

  void get() const override
  {
    std::cout << "Learning rate: " << learningRate_ << ", "
              << "Initial accumulation: " << initAccumlation_ << ", "
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

  ~adaGrad() = default;
};


}// namespace

#endif
