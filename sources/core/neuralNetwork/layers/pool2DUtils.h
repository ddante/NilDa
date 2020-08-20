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
                   pool2DDimensions& poolDims
                  );

// Get the start index of each pooling block
void getBlockHead(
                  const pool2DDimensions& dims,
                  VectorI& indices
                 );

// Compute the max value in each pooling blokc
// and return the correspondin index
void findMax(
             const pool2DDimensions& dims,
             const Scalar* colReading,
             const int col,
             int* mId,
             Scalar* out
            );

// Perform the max pooling
void maxPool2D(
               const pool2DDimensions& poolDims,
               const Matrix& input,
               Matrix& output,
               MatrixI& maxIds
              );

// Perform a sanity check
Scalar
checkPooling(
             const Matrix& input,
             const pool2DDimensions& poolDims,
             const std::string& poolType
            );


} // namespace

#endif
