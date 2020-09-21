#include <iostream>
#include "primitives/Array.h"

#include "binaryCrossentropy.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


Scalar
binaryCrossentropy::compute(
                            const Matrix& obs,
                            const Matrix& labels
                           )
{
#ifdef ND_DEBUG_CHECKS
  assert(labels.rows() == 1);
  assert(labels.cols() == obs.cols());
#endif

  int nObs = obs.cols();

  RowArray s;
  s.resize(obs.rows());

  s =        obs.array().log()  *        labels.array()
    + (1.0 - obs.array()).log() * (1.0 - labels.array());

  return (-1.0/nObs)*s.sum();
}

void
binaryCrossentropy::computeDerivative(
                                      const Matrix& obs,
                                      const Matrix& labels,
                                      Matrix& output
                                     )
{
#ifdef ND_DEBUG_CHECKS
  assert(labels.cols() == obs.cols());
#endif

  output =        -labels.array()  *        obs.array().cwiseInverse()
         +  (1.0 - labels.array()) * (1.0 - obs.array()).cwiseInverse();
}

void
binaryCrossentropy::predict(
                            const Matrix& output,
                            Matrix& predictions
                           )
{
  predictions.setZero(output.rows(), output.cols());

  Matrix ones;
  ones.setOnes(output.rows(), output.cols());

  predictions = (output.array() < 0.5).select(predictions, ones);
}

} // namespace
