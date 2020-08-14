#include <iostream>

#include <math.h>
#include <limits>

#include "conv2DUtils.h"

#include "pool2DUtils.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

const Scalar infty = std::numeric_limits<Scalar>::infinity();

void
pool2DDimensions::setDimensions(
                                const int inR,  const int inC,  const int inCh,
                                const int pT,   const int pB,
                                const int pL,   const int pR,
                                const int inPR, const int inPC,
                                const int kR,   const int kC,
                                const int kSR,  const int kSC,
                                const int outR, const int outC
                               )
{
  inputRows = inR;
  inputCols = inC;
  inputChannels = inCh;
  padTop = pT;
  padBottom = pB;
  padLeft = pL;
  padRight = pR;
  inputPaddedRows = inPR;
  inputPaddedCols = inPC;
  kernelRows = kR;
  kernelCols = kC;
  kernelChannels = inCh;
  kernelStrideRow = kSR;
  kernelStrideCol = kSC;
  outputRows = outR;
  outputCols = outC;
  outputChannels = inCh;
}

std::ostream& operator << (
                           std::ostream& os,
                           const pool2DDimensions& dims
                          )
{
  std::cout << "Input size: " << dims.inputRows << ", "
                              << dims.inputCols << ", "
                              << dims.inputChannels << std::endl;

  std::cout << "Kernel size: " << dims.kernelRows << ", "
                               << dims.kernelCols << ", "
                               << dims.kernelChannels << std::endl;

  std::cout << "Kernel stride: " << dims.kernelStrideRow << ", "
                                 << dims.kernelStrideCol << std::endl;

  std::cout << "Padding top/bottom: " << dims.padTop << ", "
                                      << dims.padBottom << std::endl;

  std::cout << "Padding left/right: " << dims.padLeft << ", "
                                      << dims.padRight << std::endl;

  std::cout << "Padded input size: " << dims.inputPaddedRows << ", "
                                     << dims.inputPaddedCols << std::endl;

  std::cout << "Output size: " << dims.outputRows << ", "
                               << dims.outputCols << std::endl;

  return os;
}

void
setPool2DDims(
              const std::array<int, 3>& inputSize,
              const std::array<int, 2>& kernelSize,
              const std::array<int, 2>& kernelStride,
              const bool withPadding,
              pool2DDimensions& poolDims
             )
{
  const int inputRows     = inputSize[0];
  const int inputCols     = inputSize[1];
  const int inputChannels = inputSize[2];

  const int kernelRows = kernelSize[0];
  const int kernelCols = kernelSize[1];

  // kernel and input channles must be equal
  const int kernelChannels = inputSize[2];

  const int kernelStrideRow = kernelStride[0];
  const int kernelStrideCol = kernelStride[1];

  assert(kernelChannels == inputChannels);
  assert(kernelRows <= inputRows);
  assert(kernelCols <= inputCols);

  // The general formula for the output size is:
  // 1 + (input - kernel + 2*pad)/kernelStride
  Scalar rows = 1 + (inputRows - kernelRows)
                 / static_cast<Scalar>(kernelStrideRow);

  Scalar cols = 1 + (inputCols - kernelCols)
                  / static_cast<Scalar>(kernelStrideCol);

  int outputRows = floor(rows);
  int outputCols = floor(cols);

  int upOutputRows = ceil(rows);
  int upOutputCols = ceil(cols);

  int padLeft = 0;
  int padRight = 0;
  int padTop = 0;
  int padBottom = 0;

  if (withPadding)
  {
    const int totalVertPad = (upOutputRows - 1) * kernelStrideRow
                           - inputRows
                           + kernelRows;

    paddingPartitioning(totalVertPad, padTop, padBottom);

    const int actualOutputRows = ceil(1
                                      + (inputRows - kernelRows + totalVertPad)
                                        / static_cast<Scalar>(kernelStrideRow)
                                     );

    assert(actualOutputRows == upOutputRows);

    const int totalHorizPad = (upOutputCols - 1) * kernelStrideCol
                            - inputCols
                            + kernelCols;

    paddingPartitioning(totalHorizPad, padLeft, padRight);

    const int actualOutputCols = ceil(1
                                      + (inputCols - kernelCols + totalHorizPad)
                                        / static_cast<Scalar>(kernelStrideCol)
                                     );

    assert(actualOutputCols == upOutputCols);

    // Update the output dimensions
    outputRows = upOutputRows;
    outputCols = upOutputCols;
}

  const int inputRowsPadded = inputRows
                            + padTop
                            + padBottom;

  const int inputColsPadded = inputCols
                            + padLeft
                            + padRight;

  poolDims.setDimensions(
                         inputRows, inputCols, inputChannels,
                         padTop, padBottom,
                         padLeft, padRight,
                         inputRowsPadded, inputColsPadded,
                         kernelRows, kernelCols,
                         kernelStrideRow, kernelStrideCol,
                         outputRows, outputCols
                        );
}

void getBlockHead(
                  const pool2DDimensions& dims,
                  VectorI& indices
                 )
{
  const int poolRows = dims.outputRows
                     * dims.outputCols;

  indices.resize(poolRows);

  const int strideRow = dims.inputPaddedCols
                      * dims.kernelStrideRow;

  int id = 0;

  for (int i = 0; i < dims.outputRows; ++i)
  {
    for (int j = 0; j < dims.outputCols; ++j, ++id)
    {
      indices[id] = (i * strideRow)
                  + (j * dims.kernelStrideCol);
    }
  }
}

void applyPoolPadding(
                      const pool2DDimensions& dims,
                      const Scalar* input,
                      const Scalar initVal,
                      RowMatrix& paddedInput
                     )
{
  const Scalar* reader = input;

  paddedInput.setOnes(
                      dims.inputPaddedRows,
                      dims.inputPaddedCols
                     );

  paddedInput *= initVal;

  Scalar* writer = paddedInput.data();

  const std::size_t copyBytes = sizeof(Scalar) * dims.inputCols;

  const int stride = dims.inputPaddedCols;

  writer += dims.inputPaddedCols * dims.padTop
          + dims.padLeft;

  for (
       int i = 0; i < dims.inputRows; ++i,
       reader += dims.inputCols,
       writer += stride
      )
  {
      std::memcpy(writer, reader, copyBytes);
  }
}

void findMax(
             const pool2DDimensions& dims,
             const Scalar* input,
             const int col,
             int* mId,
             Scalar* out
            )
{
  const int poolSize = dims.outputRows * dims.outputCols;

  const int strideRow = dims.inputRows
                      * dims.inputCols;
  for (
       int pool = 0; pool < poolSize; ++pool,
       ++mId, out++
      )
  {
    Scalar maxVal = -infty;

    int locMaxId = -1;

    const int idStart = *mId;

    for (int i = 0; i < dims.kernelRows; ++i)
    {
      for (int j = 0; j < dims.kernelCols; ++j)
      {
        const int shift = i*dims.inputPaddedCols + j;

        const int ofset = idStart + shift;

        const Scalar val = *(input + ofset);

        if(val > maxVal)
        {
          maxVal = val;

          locMaxId = ofset;

          std::cout << "i=" <<  i << ", j=" << j
                    << "p=" <<  p << ", off=" << ofset << " | "
                    << locMaxId  << ", " << val << "\n";
        }
      }
    }

    *mId = locMaxId + col*strideRow;

    *out = maxVal;
  }
}

void getMaxPool(
                const pool2DDimensions& dims,
                const Matrix& input,
                Matrix& output,
                MatrixI& maxIds
               )
{
  VectorI ofsets;

  getBlockHead(dims, ofsets);

  bool padding = false;

  if (
       dims.padLeft > 0 || dims.padRight  > 0 ||
       dims.padTop  > 0 || dims.padBottom > 0
     )
  {
      padding = true;
  }

  const int poolRows = dims.outputRows * dims.outputCols;

  const int strideCol = dims.inputRows * dims.inputCols;

  const int nCols = input.cols();

  const std::size_t copyBytes = sizeof(int) * poolRows;

  const Scalar* src = input.data();

  int* mId = maxIds.data();

  Scalar* out = output.data();

  for (
       int col = 0; col < nCols; ++col,
       src += strideCol,
       mId += poolRows,
       out += poolRows
      )
  {
    const Scalar* colReading = nullptr;

    RowMatrix padSrc;

    if (padding)
    {
      applyPoolPadding(dims, src, -infty, padSrc);

      colReading = padSrc.data();
    }
    else
    {
      colReading = src;
    }

    std::memcpy(mId, ofsets.data(), copyBytes);

    findMax(dims, colReading, col, mId, out);
  }
}

void maxPool2D(
               const pool2DDimensions& poolDims,
               const Matrix& input,
               Matrix& maxPool,
               MatrixI& maxIds
              )
{
  const int poolRows = poolDims.outputRows
                     * poolDims.outputCols;

  const int poolCols = input.cols();

  maxIds.resize(poolRows, poolCols);

  maxPool.resize(poolRows, poolCols);

  getMaxPool(poolDims, input, maxPool, maxIds);
}

} // namespace
