#include <iostream>
#include "primitives/Array.h"

#include "sparseCategoricalCrossentropy.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


Scalar
sparseCategoricalCrossentropy::compute(
                                       const Matrix& obs,
                                       const Matrix& labels
                                      )
{
#ifdef ND_DEBUG_CHECKS
  assert(labels.rows() == obs.rows());
  assert(labels.cols() == obs.cols());
#endif

  int nObs = obs.cols();

  RowArray s;
  s.resize(obs.rows());

  s = ((obs.array()).log() * labels.array()).colwise().sum()
    + ((1.0 - obs.array()).log() * (1.0 - labels.array())).colwise().sum();

  return (-1.0/nObs)*s.sum();
}

void
sparseCategoricalCrossentropy::computeDerivative(
                                                 const Matrix& obs,
                                                 const Matrix& labels,
                                                 Matrix& output
                                                )
{
#ifdef ND_DEBUG_CHECKS
  assert(labels.cols() == obs.cols());

  assert(output.rows() == obs.rows());
  assert(output.cols() == obs.cols());
#endif

  output = -labels.array() * obs.array().cwiseInverse()
         + (1.0 - labels.array()) * (1.0 - obs.array()).cwiseInverse();
}

void
sparseCategoricalCrossentropy::predict(
                                       const Matrix& output,
                                       Matrix& predictions
                                      )
{
  predictions.resize(output.rows(), output.cols());

  predictions.setZero(predictions.rows(), predictions.cols());

  const int nObs = predictions.cols();

  Matrix::Index maxIndex;

  for (int i=0; i < nObs; ++i)
  {
    Scalar maxVal = output.col(i).maxCoeff(&maxIndex);

    (void) maxVal;

    predictions(maxIndex, i) = 1;
  }
}

} // namespace
