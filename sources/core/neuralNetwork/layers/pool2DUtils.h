#ifndef POOL_2D_UTILS_H
#define POOL_2D_UTILS_H

#include <iostream>
#include <array>
#include <string>
#include <assert.h>

#include "primitives/Vector.h"
#include "primitives/Matrix.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

struct pool2DDimensions
{

public:

  int inputRows;
  int inputCols;
  int inputChannels;
  int padTop;
  int padBottom;
  int padLeft;
  int padRight;
  int inputPaddedRows;
  int inputPaddedCols;
  int kernelRows;
  int kernelCols;
  int kernelChannels;
  int kernelStrideRow;
  int kernelStrideCol;
  int outputRows;
  int outputCols;
  int outputChannels;

  pool2DDimensions() = default;

  pool2DDimensions(pool2DDimensions& other) = delete;

  // Store the dimensions for the conv2d layer
  void setDimensions(
                     const int inR,  const int inC,  const int inCh,
                     const int pT,   const int pB,
                     const int pL,   const int pR,
                     const int inPR, const int inPC,
                     const int kR,   const int kC,
                     const int kSR,  const int kSC,
                     const int outR, const int outC
                   );
};

std::ostream& operator << (
                           std::ostream& os,
                           const pool2DDimensions& dims
                          );

// Compute the dimensions of the conv2d layer
void setPool2DDims(
                   const std::array<int, 3>& inputSize,
                   const std::array<int, 2>& kernelSize,
                   const std::array<int, 2>& kernelStride,
                   const bool withPadding,
                   pool2DDimensions& poolDims
                  );


} // namespace

#endif
