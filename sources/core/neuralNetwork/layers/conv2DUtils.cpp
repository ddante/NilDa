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
    pad1 = ceil(static_cast<double>(totalPad) / 2.0);
    pad2 = totalPad - pad1;
  }
}


} // namespace
