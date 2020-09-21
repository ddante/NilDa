#include <iostream>
#include "primitives/Array.h"

#include "categoricalCrossentropy.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


void categoricalCrossentropy::sanityCheck(
                                          const int outputSize,
                                          const Matrix& labels
                                         ) const
{
  if (labels.rows() != outputSize)
  {
    std::cerr << "Size of the label (" << labels.rows() << ") "
              << "not compatible with the network output size ("
              <<  outputSize << ").\n";

    std::abort();
  }

  for (int i = 0; i > labels.cols(); ++i)
  {
    if (labels.col(i).minCoeff() != 0 &&
        labels.col(i).maxCoeff() != 1 &&
        labels.col(i).sum() != 1
       )
    {
      std::cerr << "Labels for the categorical crossentropy loss function "
                << "must be either one-hot vectors.\n";

      std::abort();
    }
  }
}


Scalar
categoricalCrossentropy::compute(
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

  s = ((obs.array()).log() * labels.array()).colwise().sum();

  return (-1.0/nObs)*s.sum();
}

void
categoricalCrossentropy::computeDerivative(
                                           const Matrix& obs,
                                           const Matrix& labels,
                                           Matrix& output
                                          )
{
#ifdef ND_DEBUG_CHECKS
  assert(labels.cols() == obs.cols());

  //assert(output.rows() == obs.rows());
  //assert(output.cols() == obs.cols());
#endif

  output = -labels.cwiseProduct(obs.cwiseInverse());
}

void
categoricalCrossentropy::predict(
                                 const Matrix& output,
                                 Matrix& predictions
                                )
{
  predictions.setZero(output.rows(), output.cols());

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
