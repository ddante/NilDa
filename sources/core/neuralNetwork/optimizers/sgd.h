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

  Scalar momentum_;

  // To recover the correct history of the weights and
  // biases associated with their respective gradient,
  // store the history in map using as key a constant pointer
  // to the gradients of the weights and biases
  mutable std::map<const Scalar*, Matrix> weightsHistory_;

  mutable std::map<const Scalar*, Vector> biasesHistory_;

public:

  // Constructors

  sgd() = delete;

  explicit sgd(Scalar alpha);

  sgd(Scalar alpha, Scalar m);

  void get() const override
  {
    std::cout << learningRate_ << " ~~~ "
                << momentum_ << "\n";
  }

  void init(
            const Matrix& weightsGradient,
            const Vector& biasesGradient
           ) const override;

  // Member functions
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
