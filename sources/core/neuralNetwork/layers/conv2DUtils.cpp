#include <iostream>

#include <math.h>

#include "conv2DUtils.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


void
conv2DDimensions::setDimensions(
                                const int inR,  const int inC,  const int inCh,
                                const int inRStride, const int inChStride,
                                const int pRL,  const int pRR,
                                const int pCT,  const int pCB,
                                const int inPR, const int inPC,
                                const int kR,   const int kC,   const int kN,
                                const int kSR,  const int kSC,
                                const int outR, const int outC
                               )
{
  inputRows = inR;
  inputCols = inC;
  padRowLeft = pRL;
  padRowRight = pRR;
  padColTop = pCT;
  padColBottom = pCB;
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

  std::cout << "Padding rows: " << dims.padRowLeft << ", "
                                << dims.padRowRight << std::endl;

  std::cout << "Padding columns: " << dims.padColTop << ", "
                                   << dims.padColBottom << std::endl;

  std::cout << "Output size: " << dims.outputRows << ", "
                               << dims.outputCols << std::endl;

  return os;
}

void paddingPartitioning(const int totalPad, int& pad1, int& pad2)
{
  // It the total padding is even, split it equally
  // between the two sides otherwise add more padding to the side_1
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
  int outputRows = floor(1 + (inputRows - filterRows) / filterStrideRow);
  int outputCols = floor(1 + (inputCols - filterCols) / filterStrideCol);

  int padRowLeft = 0;
  int padRowRight = 0;
  int padColTop = 0;
  int padColBottom = 0;

  if (withPadding)
  {
    const int totalPadRow = inputRows * (filterStrideRow - 1)
                          - filterStrideRow
                          + filterRows;

    paddingPartitioning(totalPadRow, padRowLeft, padRowRight);

    const int totalPadCol = inputCols * (filterStrideCol - 1)
                          - filterStrideCol
                          + filterCols;

    paddingPartitioning(totalPadCol, padColTop, padColBottom);

    // The output dimensions are the same as the input with padding
    outputRows = inputRows;
    outputCols = inputCols;
}

  const int inputRowsPadded = inputRows
                            + padRowLeft
                            + padRowRight;

  const int inputColsPadded = inputCols
                            + padColTop
                            + padColBottom;

  const int inputObservationStride = inputRows
                                   * inputCols
                                   * inputChannels;

  const int inputChannelStride = inputRows * inputCols;

  forwardConvDims.setDimensions(
                                inputRows, inputCols, inputChannels,
                                inputObservationStride, inputChannelStride,
                                padRowLeft, padRowRight,
                                padColTop, padColBottom,
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
                                        padRowLeft, padRowRight,
                                        padColTop, padColBottom,
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
  const int totalFullPadRow = (inputRows - 1)
                            * filterStrideRow
                            + filterRows
                            - outputRows;

  int fullPadRowLeft, fullPadRowRight;

  paddingPartitioning(
                      totalFullPadRow,
                      fullPadRowLeft,
                      fullPadRowRight
                     );

  const int outputRowsFullPadded = outputRows
                                 + fullPadRowLeft
                                 + fullPadRowRight;

  const int totalFullPadCol = (inputCols - 1)
                            * filterStrideCol
                            + filterCols
                            - outputCols;

  int fullPadColTop, fullPadColBottom;

  paddingPartitioning(
                      totalFullPadCol,
                      fullPadColTop,
                      fullPadColBottom
                     );

  const int outputColsFullPadded = outputCols
                                 + fullPadColTop
                                 + fullPadColBottom;

  // Output and input are switched with respect to the forward convolution
  backwardInputConvDims.setDimensions(
                                      outputRows, outputCols, numberOfFilters,
                                      outputObservationStrideBkw,
                                      outputChannelStrideBkw,
                                      fullPadRowLeft, fullPadRowRight,
                                      fullPadColTop, fullPadColBottom,
                                      outputRowsFullPadded, outputColsFullPadded,
                                      filterRows, filterCols, inputChannels,
                                      filterStrideRow, filterStrideCol,
                                      inputRows, inputCols
                                     );
}

Scalar* applyPadding(
                     const conv2DDimensions& dims,
                     const Scalar* input
                    )
{
    const Scalar* reader = input;

    RowMatrix paddedInput;

    paddedInput.setZero(
                        dims.inputPaddedRows,
                        dims.inputPaddedCols
                       );

    Scalar* writer = paddedInput.data();

    const std::size_t copyBytes = sizeof(Scalar) * dims.inputCols;

    const int stride = dims.inputPaddedCols;

    writer += dims.inputPaddedCols * dims.padColTop
            + dims.padRowLeft;

    for (
         int i = 0; i < dims.inputRows; ++i,
         reader += dims.inputCols, writer += stride
        )
    {
        std::memcpy(writer, reader, copyBytes);
    }

    return paddedInput.data();
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
       dims.padRowLeft > 0 || dims.padRowRight  > 0 ||
       dims.padColTop  > 0 || dims.padColBottom > 0
     )
  {
      padding = true;
  }

  for (
       int i = 0;  i < nObs; ++i,
       obs += dims.inputObservationStride
      )
  {
    const Scalar* colR  eading =
      padding ? applyPadding(dims, obs) : obs;

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

void convolve(const int nObservations,
              const conv2DDimensions& dims,
              const Scalar* input,
              const Scalar* kernels,
              Matrix& output)
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
      //std::cout << patches << "\n\n";

      ConstMapMatrix mappedKernels(
                                   kernels,
                                   kernelSize,
                                   dims.kernelNumber
                                  );

      applyConvolution(dims, patches, mappedKernels, conv);
      //std::cout << Conv << "\n\n";
  }

  output.resize(
                dims.outputRows* dims.outputCols,
                nObservations * dims.kernelNumber
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

  //std::cout << Output << "\n----\n";
}


} // namespace
