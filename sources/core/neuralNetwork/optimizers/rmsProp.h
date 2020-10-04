#ifndef rms_PROP_H
#define RMS_PROP_H

#include <iostream>
#include <memory>
#include <map>

#include "optimizer.h"

#include "primitives/Scalar.h"
#include "primitives/Array.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

class rmsProp : public optimizer
{
private:

  Scalar learningRate_;

  // Discounting factor for the history of the gradient
  Scalar decay_;

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

private:

  template <class T>
  void computeUpdate(
                     const T& gradient,
                     T& accumulator,
                     T& increment
                    ) const;
public:

  // Constructors

  rmsProp() = delete;

  rmsProp(Scalar alpha, Scalar decay);

  rmsProp(
          Scalar alpha,
          Scalar decay,
          Scalar initAccumlation
         );

  rmsProp(
          Scalar alpha,
          Scalar decay,
          Scalar initAccumlation,
          Scalar epsilon
         );

  // Member functions

  void get() const override
  {
    std::cout << "Learning rate: " << learningRate_ << ", "
              << "Decay: " << decay_ << ", "
              << "Initial momentum: " << initAccumlation_ << ", "
              << "Epsilon: " << epsilon_ << "\n";
  }

  void init(
            const Matrix& weightsGradient,
            const Vector& biasesGradient
           ) const override;


  void update(
              const Matrix& weightsGradient,
              const Vector& biasesGradient,
              Matrix& deltaWeights,
              Vector& deltaBiases
             ) const override;

  // Destructor

  ~rmsProp() = default;
};


}// namespace

#endif
