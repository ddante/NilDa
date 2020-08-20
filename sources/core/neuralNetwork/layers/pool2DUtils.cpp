#include <iostream>

#include <math.h>
#include <algorithm>
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
                                const int kR,   const int kC,
                                const int kSR,  const int kSC,
                                const int outR, const int outC
                               )
{
  inputRows = inR;
  inputCols = inC;
  inputChannels = inCh;
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

  std::cout << "Output size: " << dims.outputRows << ", "
                               << dims.outputCols << std::endl;

  return os;
}

void
setPool2DDims(
              const std::array<int, 3>& inputSize,
              const std::array<int, 2>& kernelSize,
              const std::array<int, 2>& kernelStride,
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
  // 1 + (input - kernel + 2*pad)/kernelStride.
  // Round the number down.
  const int outputRows = floor(
                               1 + (inputRows - kernelRows)
                                 / static_cast<Scalar>(kernelStrideRow)
                              );

  const int outputCols = floor(
                               1 + (inputCols - kernelCols)
                                 / static_cast<Scalar>(kernelStrideCol)
                              );

  poolDims.setDimensions(
                         inputRows, inputCols, inputChannels,
                         kernelRows, kernelCols,
                         kernelStrideRow, kernelStrideCol,
                         outputRows, outputCols
                        );
}

void getBlockHead(
                  const pool2DDimensions& dims,
                  MatrixI& indices
                 )
{
#ifdef ND_DEBUG_CHECKS
  assert(indices.rows() == dims.outputRows * dims.outputCols);
#endif

  const int strideRow = dims.inputCols
                      * dims.kernelStrideRow;

  const int strideInput = dims.inputRows
                        * dims.inputCols;

  const int nCols = indices.cols();

  int* id = indices.data();

  for (int col = 0; col < nCols; ++col)
  {
    for (int i = 0; i < dims.outputRows; ++i)
    {
      for (int j = 0; j < dims.outputCols; ++j, ++id)
      {
        *id = (i * strideRow)
            + (j * dims.kernelStrideCol)
            + (col * strideInput);
      }
    }
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

  // For each block, start from the first index (mId) and
  // look for the max value in the neighbour cols and rows.
  // The max value is stroed in out and the correspondid index
  // is store in mId
  for (
       int pool = 0; pool < poolSize; ++pool,
       ++mId, ++out
      )
  {
    Scalar maxVal = -infty;

    int maxId = -1;

    // mId contains the index of the first element of the block.
    // Because it is update, this index must be store in a
    // separate variable (idStart)
    const int idStart = *mId;

    for (int i = 0; i < dims.kernelRows; ++i)
    {
      for (int j = 0; j < dims.kernelCols; ++j)
      {
        // shift is local index shift w.r.t the first
        // index of the BLOCK
        const int shift = i*dims.inputCols + j;

        // ofset is global index of the element in the INPUT
        const int ofset = idStart + shift;

        const Scalar val = *(input + ofset);

        if(val > maxVal)
        {
          maxVal = val;

          maxId = ofset;
        }
      }
    }

    *mId = maxId;

    *out = maxVal;
  }
}

void maxPool2D(
               const pool2DDimensions& poolDims,
               const Matrix& input,
               Matrix& maxPool,
               MatrixI& maxIds
              )
{
  const int nCols = input.cols();

  const int poolSize = poolDims.outputRows
                     * poolDims.outputCols;

  maxIds.resize(poolSize, nCols);

  // Store the indices of the first element of the pool
  // block in maxIds
  getBlockHead(poolDims, maxIds);

  const Scalar* src = input.data();

  int* mId = maxIds.data();

  maxPool.resize(poolSize, nCols);

#ifdef ND_DEBUG_CHECKS
  assert(maxPool.rows() == maxIds.rows());
  assert(maxPool.cols() == maxIds.cols());
#endif

  Scalar* out = maxPool.data();

  // Find the max values in each pool block and store
  // the corrspondidn index in maxIds
  for (
       int col = 0; col < input.cols(); ++col,
       mId += poolSize,
       out += poolSize
      )
  {
    findMax(poolDims, src, col, mId, out);
  }
}

Scalar
checkPooling(
             const Matrix& input,
             const pool2DDimensions& poolDims,
             const std::string& poolType
            )
{
  const int strideRow = poolDims.inputCols
                      * poolDims.kernelStrideRow;

  const int strideInput = poolDims.inputRows
                        * poolDims.inputCols;

  Matrix maxPool;

  MatrixI maxIds;

  maxPool2D(poolDims, input, maxPool, maxIds);

  Matrix maxPoolCheck;
  maxPoolCheck.setZero(maxPool.rows(), maxPool.cols());

  MatrixI maxIdsCheck;
  maxIdsCheck.setZero(maxIds.rows(), maxIds.cols());

  for (int col = 0; col < input.cols(); ++col)
  {
    Eigen::Map<const RowMatrix> A(
                                  input.col(col).data(),
                                  poolDims.inputRows,
                                  poolDims.inputCols
                                 );
    int l = 0;
    for (int i = 0; i < poolDims.outputRows; ++i)
    {
      for (int j = 0; j < poolDims.outputCols; ++j, ++l)
      {
        Matrix tmp = A.block(
                             i*poolDims.kernelStrideRow,
                             j*poolDims.kernelStrideCol,
                             poolDims.kernelRows,
                             poolDims.kernelCols
                            );
         if (poolType == "max")
         {
           Matrix::Index maxRow, maxCol;
           Scalar max = tmp.maxCoeff(&maxRow, &maxCol);

           const int loc = i * strideRow
                         + maxRow * poolDims.inputCols
                         + j * poolDims.kernelStrideCol
                         + maxCol
                         + col * strideInput;

           maxPoolCheck(l, col) = max;

           maxIdsCheck(l, col) = loc;
         }
         else
         {
           std::cerr << "Unknown pooling operation "
                     << poolType << std::endl;

           return infty;

           assert(false);
         }
      }
    }
  }

  Scalar err1 = (maxPoolCheck - maxPool).array().abs().mean();

  Scalar err2 = (maxIdsCheck - maxIds).array().abs().mean();

  return std::max(err1, err2);
}

} // namespace
