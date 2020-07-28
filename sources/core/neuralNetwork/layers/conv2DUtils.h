#ifndef CONV_2D_UTILS_H
#define CONV_2D_UTILS_H

#include <iostream>

// ---------------------------------------------------------------------------

namespace NilDa
{

class conv2DDimensions
{

private:

  int inputRows;
  int inputCols;
  int padRowLeft;
  int padRowRight;
  int padColTop;
  int padColBottom;
  int inputPaddedRows;
  int inputPaddedCols;
  int inputChannels;
  int inputObservationStride;
  int inputChannelStride;
  int kernelRows;
  int kernelCols;
  int kernelChannels;
  int kernelNumber;
  int kernelStrideRow;
  int kernelStrideCol;
  int outputRows;
  int outputCols;
  int outputChannels;

private:

  conv2DDimensions() = default;

  conv2DDimensions(conv2DDimensions& other) = delete;

  void setDimensions(
                     const int inR,  const int inC,  const int inCh,
                     const int inRStride, const int inChStride,
                     const int pRL,  const int pRR,
                     const int pCT,  const int pCB,
                     const int inPR, const int inPC,
                     const int kR,   const int kC,   const int kN,
                     const int kSR,  const int kSC,
                     const int outR, const int outC
                   );

  // This is method is needed for the convolve function
  // in the backward propagation because the number
  // of channels of the input becomes the number of observations
  void setInputChannels(const int inCh);

  friend class conv2DLayer;

  friend std::ostream& operator << (
                                    std::ostream& os,
                                    const conv2DDimensions& dims
                                   );
};

std::ostream& operator << (
                           std::ostream& os,
                           const conv2DDimensions& dims
                          );

void paddingPartitioning(const int totalPad, int& pad1, int& pad2);

} // namespace

#endif
