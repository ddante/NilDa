#include <iostream>
#include <math.h>

#include "primitives/Array.h"

#include "sparseCategoricalCrossentropy.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


void
sparseCategoricalCrossentropy::sanityCheck(
                                           const int outputSize,
                                           const Matrix& labels
                                          ) const
{
  if (labels.rows() != 1)
  {
    std::cerr << "Sparse crossentropy loss function requires "
              << "one single label per data.\n";

    std::abort();
  }

  for (int i = 0; i > labels.cols(); ++i)
  {
    if (labels.col(i).minCoeff() < 0           &&
        labels.col(i).maxCoeff() >= outputSize
       )
    {
      std::cerr << "Incorrect labels for the sparse categorical "
                << "crossentropy loss function.\n";

      std::abort();
    }
  }
}

Scalar
sparseCategoricalCrossentropy::compute(
                                       const Matrix& obs,
                                       const Matrix& labels
                                      )
{
#ifdef ND_DEBUG_CHECKS
  assert(labels.rows() == 1);
  assert(labels.cols() == obs.cols());
#endif

  int nObs = obs.cols();

  //RowArrayI sparseIndices = labels.row(0).cast<int>();

  Scalar s = 0;

  for (int i = 0; i < nObs; ++i)
  {
    const int idx = labels(0, i);

    s += log(obs(idx, i));
  }

  return (-1.0/nObs)*s;
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
  assert(labels.rows() == 1);

  //assert(output.rows() == obs.rows());
  //assert(output.cols() == obs.cols());
#endif

  output.setZero(obs.rows(), obs.cols());

  int nObs = obs.cols();

  for (int i = 0; i < nObs; ++i)
  {
    const int idx = labels(0, i);
    
    if(obs(idx, i) == 0)
    {
        std::cout << "\n" << i << ", " << idx << "\n";
        std::cout << obs.col(i) << "\n";
        assert(false);
    }

    output(idx, i) = -1.0/obs(idx, i);
  }

  //output = -labels.array() * obs.array().cwiseInverse();
}

void
sparseCategoricalCrossentropy::predict(
                                       const Matrix& output,
                                       Matrix& predictions
                                      )
{
  predictions.setZero(1, output.cols());

  const int nObs = predictions.cols();

  for (int i=0; i < nObs; ++i)
  {
    Matrix::Index maxIndex;

    Scalar maxVal = output.col(i).maxCoeff(&maxIndex);

    (void) maxVal;

    predictions(0, i) = maxIndex;
  }
}

} // namespace
