#include <iostream>

#include <math.h>

#include "conv2DUtils.h"

#include "primitives/Memory.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


void
conv2DDimensions::setDimensions(
                                const int inR,  const int inC,  const int inCh,
                                const int inRStride, const int inChStride,
                                const int pT,  const int pB,
                                const int pL,  const int pR,
                                const int inPR, const int inPC,
                                const int kR,   const int kC,   const int kN,
                                const int kSR,  const int kSC,
                                const int outR, const int outC
                               )
{
  inputRows = inR;
  inputCols = inC;
  padTop = pT;
  padBottom = pB;
  padLeft = pL;
  padRight = pR;
  inputPaddedRows = inPR;
  inputPaddedCols = inPC;
  inputChannels = inCh;
  inputObservationStride = inRStride;
  inputChannelStride = inChStride;
  kernelRows = kR;
  kernelCols = kC;
  kernelChannels = inCh;
  kernelNumber = kN;
  kernelStrideRow = kSR;
  kernelStrideCol = kSC;
  outputRows = outR;
  outputCols = outC;
  outputChannels = kN;
}

void conv2DDimensions::setInputChannels(const int inCh)
{
    inputChannels = inCh;
}

std::ostream& operator << (
                           std::ostream& os,
                           const conv2DDimensions& dims
                          )
{
  std::cout << "Input size: " << dims.inputRows << ", "
                              << dims.inputCols << ", "
                              << dims.inputChannels << std::endl;

  std::cout << "Filter size: " << dims.kernelRows << ", "
                               << dims.kernelCols << ", "
                               << dims.kernelChannels << std::endl;

  std::cout << "Number of filters: " << dims.kernelNumber << std::endl;

  std::cout << "Filter stride: " << dims.kernelStrideRow<< ", "
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

void paddingPartitioning(const int totalPad, int& pad1, int& pad2)
{
  // If the total padding is even, split it equally
  // between the two sides otherwise add more padding to side_1
  if (totalPad % 2 == 0)
  {
    pad1 = totalPad / 2;
    pad2 = totalPad / 2;
  }
  else
  {
    pad1 = ceil(static_cast<Scalar>(totalPad) / 2.0);
    pad2 = totalPad - pad1;
  }
}

void
setConv2DDims(
              const std::array<int, 3>& inputSize,
              const int numberOfFilters,
              const std::array<int, 2>& filterSize,
              const std::array<int, 2>& filterStride,
              const bool withPadding,
              conv2DDimensions& forwardConvDims,
              conv2DDimensions& backwardWeightsConvDims,
              conv2DDimensions& backwardInputConvDims
             )
{
  const int inputRows     = inputSize[0];
  const int inputCols     = inputSize[1];
  const int inputChannels = inputSize[2];

  const int filterRows = filterSize[0];
  const int filterCols = filterSize[1];

  // Filter and input channles must be equal
  const int filterChannels = inputSize[2];

  const int filterStrideRow = filterStride[0];
  const int filterStrideCol = filterStride[1];

  assert(filterChannels == inputChannels);
  assert(filterRows <= inputRows);
  assert(filterCols <= inputCols);

  // --- Dimensions to perform the convolution of input with filter ---

  // The general formula for the output size is:
  // 1 + (input - filter + 2*pad)/filterStride
  int outputRows = floor(
                         1 + (inputRows - filterRows)
                           / static_cast<Scalar>(filterStrideRow)
                        );


  int outputCols = floor(
                         1 + (inputCols - filterCols)
                           / static_cast<Scalar>(filterStrideCol)
                         );

  int padLeft = 0;
  int padRight = 0;
  int padTop = 0;
  int padBottom = 0;

  if (withPadding)
  {
    const int totalVertPad = inputRows * (filterStrideRow - 1)
                           - filterStrideRow
                           + filterRows;

    // Compute the resulting output rows with the total padding
    const int actualOutputRows = ceil(
                                      1
                                      + (inputRows - filterRows + totalVertPad)
                                        / static_cast<Scalar>(filterStrideRow)
                                     );

    assert(actualOutputRows == inputRows);

    paddingPartitioning(totalVertPad, padTop, padBottom);

    const int totalHorizPad = inputCols * (filterStrideCol - 1)
                            - filterStrideCol
                            + filterCols;

    // Compute the resulting output cols with the total padding
    const int actualOutputCols = ceil(
                                      1
                                      + (inputCols - filterCols + totalHorizPad)
                                        / static_cast<Scalar>(filterStrideCol)
                                     );

    assert(actualOutputCols == inputCols);

    paddingPartitioning(totalHorizPad, padLeft, padRight);

    // The output dimensions are the same as the input with padding
    outputRows = actualOutputRows;
    outputCols = actualOutputCols;
}

  const int inputRowsPadded = inputRows
                            + padTop
                            + padBottom;

  const int inputColsPadded = inputCols
                            + padLeft
                            + padRight;

  const int inputObservationStride = inputRows
                                   * inputCols
                                   * inputChannels;

  const int inputChannelStride = inputRows * inputCols;

  forwardConvDims.setDimensions(
                                inputRows, inputCols, inputChannels,
                                inputObservationStride, inputChannelStride,
                                padTop, padBottom,
                                padLeft, padRight,
                                inputRowsPadded, inputColsPadded,
                                filterRows, filterCols, numberOfFilters,
                                filterStrideRow, filterStrideCol,
                                outputRows, outputCols
                               );

  //  ---------------------------------------------------------


  // --- Dimensions to perform the convolution of input with output ---

  const int inputObservationStrideBkw = inputRows * inputCols;

  const int inputChannelStrideBkw = inputRows
                                  * inputCols
                                  * inputChannels;

  // The number of input channels is set to 0 because
  // it will be replaced by the number of observations
  backwardWeightsConvDims.setDimensions(
                                        inputRows, inputCols, 0,
                                        inputObservationStrideBkw,
                                        inputChannelStrideBkw,
                                        padTop, padBottom,
                                        padLeft, padLeft,
                                        inputRowsPadded, inputColsPadded,
                                        outputRows, outputCols, numberOfFilters,
                                        filterStrideRow, filterStrideCol,
                                        filterRows, filterCols
                                       );


  //  ---------------------------------------------------------

  // -- Dimensions to perform the convolution of output with filter ---

  const int outputObservationStrideBkw = outputRows
                                       * outputCols
                                       * numberOfFilters;

  const int outputChannelStrideBkw = outputRows * outputCols;

  // Padding such that the result of the convolution has the dimension
  // of the original input of the forward convolution
  const int totalVertPad = (inputRows - 1) * filterStrideRow
                         + filterRows
                         - outputRows;

  int fullPadTop, fullPadBottom;

  // Note that the order of Top and Bottom padding are switched.
  // For asymmetric padding, the padding on the two sides
  // must be reverted compared to the forward step
  // (No idea why)
  paddingPartitioning(
                      totalVertPad,
                      fullPadBottom,
                      fullPadTop
                     );

  const int outputRowsFullPadded = outputRows
                                 + fullPadTop
                                 + fullPadBottom;

  const int actualInputRows = ceil(
                                   1
                                   + (outputRows - filterRows + totalVertPad)
                                     / static_cast<Scalar>(filterStrideRow)
                                  );

  assert(actualInputRows == inputRows);

  const int totalHorizPad = (inputCols - 1) * filterStrideCol
                          + filterCols
                          - outputCols;

  int fullPadLeft, fullPadRight;

  // Note that the order of Left and Right padding are switched.
  // For asymmetric padding, the padding on the two sides
  // must be reverted compared to the forward step
  // (No idea why)
  paddingPartitioning(
                      totalHorizPad,
                      fullPadRight,
                      fullPadLeft
                     );

  const int outputColsFullPadded = outputCols
                                 + fullPadLeft
                                 + fullPadRight;

  const int actualInputCols = ceil(
                                   1
                                   + (outputCols - filterCols + totalHorizPad)
                                     / static_cast<Scalar>(filterStrideCol)
                                  );

  assert(actualInputCols == inputCols);

  // Output and input are switched with respect to the forward convolution
  backwardInputConvDims.setDimensions(
                                      outputRows, outputCols, numberOfFilters,
                                      outputObservationStrideBkw,
                                      outputChannelStrideBkw,
                                      fullPadTop, fullPadBottom,
                                      fullPadLeft, fullPadRight,
                                      outputRowsFullPadded, outputColsFullPadded,
                                      filterRows, filterCols, inputChannels,
                                      filterStrideRow, filterStrideCol,
                                      inputRows, inputCols
                                     );
}

//Scalar*
void applyPadding(
                     const conv2DDimensions& dims,
                     const Scalar* input,
                     RowMatrix& paddedInput
                    )
{
  const Scalar* reader = input;

  paddedInput.setZero(
                      dims.inputPaddedRows,
                      dims.inputPaddedCols
                     );

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

void extractPatches(
                    const int nObs,
                    const conv2DDimensions& dims,
						        const Scalar* obs,
							      RowMatrix& mecMat
                   )
{
  Scalar* writer = mecMat.data();

  const int inputSize = dims.inputPaddedRows
                      * dims.inputPaddedCols;

  const int segmentSize = dims.kernelCols;

  const std::size_t copyBytes = sizeof(Scalar) * segmentSize;

  bool padding = false;

  if (
       dims.padLeft > 0 || dims.padRight  > 0 ||
       dims.padTop  > 0 || dims.padBottom > 0
     )
  {
      padding = true;
  }

  for (
       int i = 0;  i < nObs; ++i,
       obs += dims.inputObservationStride
      )
  {
    const Scalar* colReading = nullptr;

    RowMatrix padObs;

    if (padding)
    {
      applyPadding(dims, obs, padObs);

      colReading = padObs.data();
    }
    else
    {
      colReading = obs;
    }

    for (int j = 0; j < dims.outputCols; ++j)
  	{
  		const Scalar* reader = colReading + (dims.kernelStrideCol*j);

  		const Scalar* const readerEnd = reader + inputSize;

  		for (
           ; reader < readerEnd; reader += dims.inputPaddedCols,
           writer += segmentSize
          )
  		{
  			std::memcpy(writer, reader, copyBytes);
  		}
  	}
  }
}

void applyConvolution(
                      const conv2DDimensions& dims,
                      const RowMatrix& patches,
		                  const Matrix& kernels,
		                  Matrix& conv
                     )
{
  /*
  conv[:, 1:nFilters], Conv[:, 1:nFilters] ... Conv[:, 1:nFilters]
  */

  const int rowsPatches = patches.rows();
  const int colsPatches = patches.cols();

  const int rowsKernelMap = kernels.rows();
  const int colsKernelMap = kernels.cols();

  const int endColPatches = colsPatches - rowsKernelMap; //--> outputRows

  int colConv = 0;

  for (
       int slidingCol = 0;
       slidingCol <= endColPatches;
       slidingCol += dims.kernelCols * dims.kernelStrideRow,
  	   colConv += colsKernelMap
      )
  {
  	conv.block(
               0,
               colConv,
               rowsPatches,
               colsKernelMap
              ).noalias() += patches.block(
                                           0,
                                           slidingCol,
                                           rowsPatches,
                                           rowsKernelMap
                                          ) * kernels;
  }
}

void convolve(
              const int nObservations,
              const conv2DDimensions& dims,
              const Scalar* input,
              const Scalar* kernels,
              Matrix& output
             )
{
  /*
  Input Shape:
  |I1 ch1|I1 ch2||I1 ch3||I2 ch1|I2 ch2||I2 ch3|...|In ch1|In ch2||In ch3|
  {---------------------}{---------------------}   {---------------------}
          image 1                image 2                   image n

  Filter Shape (kernel):
  |f1 ch1|f2 ch1|...|fm ch1||f1 ch2|f2 ch2|...|fm ch2||f1 ch3|f2 ch3|...|fm ch3|
  {------------------------}{------------------------}{------------------------}
           in channel 1             in channel 2             in channel 3

  Output Shape:
  |O1 ch1|O1 ch2|...|O1 chm||O2 ch1|O2 ch2|...|O2 chm|...|On ch1|On ch2|...|On chm|
  {------------------------}{------------------------}   {------------------------}
           image 1                  image 2                      image n
  */

  const int kernelSize = dims.kernelRows * dims.kernelCols;

  const int kernelReadStride = kernelSize * dims.kernelNumber;

  RowMatrix patches(
                    dims.outputCols*nObservations,
                    dims.kernelCols*dims.inputPaddedRows
                   );

  Matrix conv = Matrix::Zero(
                             patches.rows(),
                             dims.outputRows*dims.kernelNumber
                            );

  for (
       int i = 0; i < dims.inputChannels; ++i,
       input += dims.inputChannelStride,
       kernels += kernelReadStride
      )
  {
      extractPatches(nObservations, dims, input, patches);
      // std::cout << patches << "\n\n";

      ConstMapMatrix mappedKernels(
                                   kernels,
                                   kernelSize,
                                   dims.kernelNumber
                                  );

      applyConvolution(dims, patches, mappedKernels, conv);
      // std::cout << conv << "\n\n";
  }

  output.resize(
                dims.outputRows* dims.outputCols,
                dims.kernelNumber * nObservations
               );

  Scalar* writer = output.data();

  const Scalar* convData = conv.data();

  const std::size_t copyBytes = sizeof(Scalar) * dims.outputCols;

  for (
       int i = 0; i < nObservations; ++ i,
       convData += dims.outputCols
      )
  {
      for (int j = 0; j < dims.kernelNumber; ++j)
      {
          for (
               int k = 0; k < dims.outputRows; ++k,
               writer += dims.outputCols
              )
          {
              const int strideConv = nObservations
                                   * dims.outputCols
                                   * (dims.kernelNumber*k + j);

              std::memcpy(writer, convData + strideConv, copyBytes);
          }
     }
  }
  //std::cout << output << "\n----\n";
}

void applyRotation(
                   const Matrix& kernels,
                   const conv2DDimensions& dims,
                   Matrix& rotatedKernels
                  )
{
  rotatedKernels.setZero(
                         kernels.rows(),
                         kernels.cols()
                        );

  const int kernelTotChannels = dims.inputChannels
                              * dims.kernelNumber;

  const int kernelSize = dims.kernelRows
                       * dims.kernelCols;

  const Scalar* reader = kernels.data();

  // Rotate the kernels and store it in a a new arrangement
  // because the number of filters and the input channels are switched
  //
  // Original arrangement:
  // |f1 ch1|f2 ch1|...|fm ch1||f1 ch2|f2 ch2|...|fm ch2||f1 ch3|f2 ch3|...|fm ch3|
  // {------------------------}{------------------------}{------------------------}
  //          in channel 1             in channel 2              in channel 3
  //
  // New arrangement:
  // |f1 ch1|f1 ch2||f1 ch3||f2 ch1|f2 ch2||f2 ch3|...|fm ch1|fm ch2||fm ch3|
  // {---------------------}{---------------------}...{---------------------}
  //         filter 1              filter 2                   filter 3
  //

  for (
       int i = 0; i <  kernelTotChannels; ++i,
       reader += kernelSize
      )
  {
      // To be refactored
      const int stride = floor(i / dims.kernelNumber)
                       + (
                          (i % dims.kernelNumber)
                          * dims.inputChannels
                         );

      Scalar* writer = rotatedKernels.data()
                     + (stride * kernelSize);

      // WTF! std::reverse_copy does not work in release mode,
      // so a special function has been implemented
      memcpy_reverse(writer,
                     reader,
                     reader + kernelSize);
  }

}

Scalar
checkConvolution(
                 const Matrix& input,
                 const Matrix& kernels,
                 const conv2DDimensions& forwardConvDims
                )
{
  const int nObs = input.cols()
                 / forwardConvDims.inputChannels;

  Matrix origConv;

  // MEC conv2D algorithm
  convolve(
           nObs,
           forwardConvDims,
           input.data(),
           kernels.data(),
           origConv
          );

  // Brute force convolution
  int colOut = 0;

  Scalar error = 0;

  for (int nO = 0; nO < nObs; ++nO)
  {
    for (int chO = 0; chO < forwardConvDims.kernelNumber; ++ chO)
    {
      RowMatrix C;
      C.setZero(
                forwardConvDims.outputRows,
                forwardConvDims.outputCols
               );

      int nColF = chO;
      int nColI = nO*forwardConvDims.inputChannels;

      for (int chI = 0; chI < forwardConvDims.inputChannels; ++chI,
           ++nColI, nColF += forwardConvDims.kernelNumber)
      {
        Eigen::Map<const RowMatrix> tmp(
                                        input.col(nColI).data(),
                                        forwardConvDims.inputRows,
                                        forwardConvDims.inputCols
                                       );

        Matrix A;
        A.setZero(
                  forwardConvDims.inputPaddedRows,
                  forwardConvDims.inputPaddedCols
                 );

        A.block(
                forwardConvDims.padTop,
                forwardConvDims.padLeft,
                forwardConvDims.inputRows,
                forwardConvDims.inputCols
               ) = tmp;

        Eigen::Map<const RowMatrix> W(
                                      kernels.col(nColF).data(),
                                      forwardConvDims.kernelRows,
                                      forwardConvDims.kernelCols
                                     );

        for (int i = 0; i < forwardConvDims.outputRows; ++i)
        {
          for (int j = 0; j < forwardConvDims.outputCols; ++j)
          {

            Matrix tmp = A.block(
                                 i*forwardConvDims.kernelStrideRow,
                                 j*forwardConvDims.kernelStrideCol,
                                 forwardConvDims.kernelRows,
                                 forwardConvDims.kernelCols
                                ).array() * W.array();

            C(i, j) += tmp.sum();
          }
        }
      }

      Eigen::Map<Matrix> tmp(
                             C.data(),
                             forwardConvDims.outputRows * forwardConvDims.outputCols,
                             1
                            );

      error += (origConv.col(colOut) - tmp).array().abs().sum();

      colOut++;
    }
  }

  return error;
}


} // namespace
