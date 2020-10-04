#ifndef SGD_H
#define SGD_H

#include <iostream>
#include <memory>
#include <map>

#include "optimizer.h"

#include "primitives/Scalar.h"
#include "primitives/Array.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

class sgd : public optimizer
{

private:

  Scalar learningRate_;

  // Hperparameter that accelerates gradient descent
  Scalar momentum_;

  // To recover the correct history of the weights and
  // biases associated with their respective gradient,
  // store the history in map using as key a constant pointer
  // to the gradients of the weights and biases
  mutable std::map<const Scalar*, Matrix> weightsHistory_;

  mutable std::map<const Scalar*, Vector> biasesHistory_;

private:

  template <class T>
  void computeUpdate(
                     const T& gradient,
                     T& firstMomentum,
                     T& increment
                    ) const;

public:

  // Constructors

  sgd() = delete;

  explicit sgd(Scalar alpha);

  sgd(Scalar alpha, Scalar m);

  // Member functions

  void get() const override
  {
    std::cout << "Learning rate: " << learningRate_ << ", "
              << "Momentum: " << momentum_ << "\n";
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

  ~sgd() = default;

};


} // namespace

#endif
