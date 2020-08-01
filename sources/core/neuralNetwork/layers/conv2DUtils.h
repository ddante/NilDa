#ifndef CONV_2D_UTILS_H
#define CONV_2D_UTILS_H

#include <iostream>
#include <array>
#include <string>
#include <assert.h>

#include "primitives/Vector.h"
#include "primitives/Matrix.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

struct conv2DDimensions
{

public:

  int inputRows;
  int inputCols;
  int padTop;
  int padBottom;
  int padLeft;
  int padRight;
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

  conv2DDimensions() = default;

  conv2DDimensions(conv2DDimensions& other) = delete;

  // Store the dimensions for the conv2d layer
  void setDimensions(
                     const int inR,  const int inC,  const int inCh,
                     const int inRStride, const int inChStride,
                     const int pT,  const int pB,
                     const int pL,  const int pR,
                     const int inPR, const int inPC,
                     const int kR,   const int kC,   const int kN,
                     const int kSR,  const int kSC,
                     const int outR, const int outC
                   );

  // This is method is needed for the convolve function
  // in the backward propagation because the number
  // of channels of the input becomes the number of observations
  void setInputChannels(const int inCh);
};

std::ostream& operator << (
                           std::ostream& os,
                           const conv2DDimensions& dims
                          );

// Compute the ammount of padding  on the two sides of the matrix
void paddingPartitioning(const int totalPad, int& pad1, int& pad2);

// Compute the dimensions of the conv2d layer
void setConv2DDims(
                   const std::array<int, 3>& inputSize,
                   const int numberOfFilters,
                   const std::array<int, 2>& filterSize,
                   const std::array<int, 2>& filterStride,
                   const bool withPadding,
                   conv2DDimensions& forwardConvDims,
                   conv2DDimensions& backwardWeightsConvDims,
                   conv2DDimensions& backwardInputConvDims
                  );

// Convolution of the input with the kernel
void convolve(
              const int nObservations,
              const conv2DDimensions& dims,
              const Scalar* Input,
              const Scalar* Kernels,
              Matrix& Output
             );

// Apply the padding to input according to the rules in dims
Scalar* applyPadding(
                     const conv2DDimensions& dims,
                     const Scalar* input
                    );

// Extract the patches for the computation of the convolution
// using a modified im2col algorithm
void extractPatches(
                    const int nObs,
                    const conv2DDimensions& dims,
						  	    const Scalar* obs,
								    RowMatrix& mecMat
                   );

// Apply the convolution by multiplying the matrix of patches
// with the kernel of the fileters
void applyConvolution(
                      const conv2DDimensions& dims,
                      const RowMatrix& patches,
		                  const Matrix& kernels,
		                  Matrix& conv
                     );

// Check the MEC conv2d algorithm with the naive
// implementation and return the difference between
// the results of the two implementations
Scalar
checkConvolution(
                 const Matrix& input,
                 const Matrix& kernels,
                 const conv2DDimensions& forwardConvDims
                );


} // namespace

#endif
