#include <iostream>

#include "conv2DLayer.h"

#include "conv2DUtils.h"

#include "activationFunctions/activationFunctionUtils.h"

#include "activationFunctions/identity.h"
#include "activationFunctions/sigmoid.h"
#include "activationFunctions/relu.h"
#include "activationFunctions/softmax.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

conv2DLayer::conv2DLayer(
                         const int numberOfFilters,
                         const std::array<int, 2>& filterSize,
                         const std::array<int, 2>& filterStride,
                         const bool withPadding,
                         const std::string& activationName
                       ) :
  numberOfFilters_(numberOfFilters),
  filterSize_(filterSize),
  filterStride_(filterStride),
  withPadding_(withPadding),
  activationName_(activationName),
  forwardConvDims_{},
  backwardWeightsConvDims_{}
{
  type_ = layerTypes::dense;

  trainable_ = true;
}

conv2DLayer::conv2DLayer(
                         const int numberOfFilters,
                         const std::array<int, 2>& filterSize,
                         const std::array<int, 2>& filterStride,
                         const std::string& activationName
                        ) :
  numberOfFilters_(numberOfFilters),
  filterSize_(filterSize),
  filterStride_(filterStride),
  withPadding_(false),
  activationName_(activationName),
  forwardConvDims_{},
  backwardWeightsConvDims_{}
{
  type_ = layerTypes::dense;

  trainable_ = true;
}

conv2DLayer::conv2DLayer(
                         const int numberOfFilters,
                         const std::array<int, 2>& filterSize,
                         const bool withPadding,
                         const std::string& activationName
                        ) :
  numberOfFilters_(numberOfFilters),
  filterSize_(filterSize),
  filterStride_({1,1}),
  withPadding_(withPadding),
  activationName_(activationName),
  forwardConvDims_{},
  backwardWeightsConvDims_{}
{
  type_ = layerTypes::dense;

  trainable_ = true;
}

conv2DLayer::conv2DLayer(
                         const int numberOfFilters,
                         const std::array<int, 2>& filterSize,
                         const std::string& activationName
                        ) :
  numberOfFilters_(numberOfFilters),
  filterSize_(filterSize),
  filterStride_({1,1}),
  withPadding_(false),
  activationName_(activationName),
  forwardConvDims_{},
  backwardWeightsConvDims_{}
{
  type_ = layerTypes::dense;

  trainable_ = true;
}

void conv2DLayer::init(const layer* previousLayer)
{
  switch (activationFunctionCode(activationName_))
  {
    case activationFucntions::identity :
      activationFunction_ = std::make_unique<identity>();
      break;
    case activationFucntions::sigmoid :
      activationFunction_ = std::make_unique<sigmoid>();
      break;
    case activationFucntions::relu :
      activationFunction_ = std::make_unique<relu>();
      break;
    default :
     std::cerr << "Not valid activation function  "
               << activationName_
               << " in this context." << std::endl;
     assert(false);
   }

   if
   (
    previousLayer->layerType() != layerTypes::input &&
    previousLayer->layerType() != layerTypes::conv2D
   )
   {
     std::cerr << "Previous layer of type "
               <<  layerName(previousLayer->layerType())
               << " not compatible with current layer of type "
               << layerName(type_) << "." << std::endl;

     assert(false);
   }

   std::array<int, 3> prevLayerSize;
   previousLayer->size(prevLayerSize);

   assert(prevLayerSize[0] > 0);
   assert(prevLayerSize[1] > 0);
   assert(prevLayerSize[2] > 0);

   setConv2DDims(prevLayerSize);
}

void conv2DLayer::checkInputSize(const Matrix& inputData) const
{}

void conv2DLayer::forwardPropagation(const Matrix& inputData)
{}

void conv2DLayer::backwardPropagation(
                                      const Matrix& dActivationNex,
                                      const Matrix& inputData
                                     )
{}

void conv2DLayer::setWeightsAndBiases(
                                      const Matrix& W,
                                      const Vector& b
                                     )
{}

void conv2DLayer::incrementWeightsAndBiases(
                                            const Matrix& deltaW,
                                            const Vector& deltaB
                                           )
{}

void
conv2DLayer::setConv2DDims(const std::array<int, 3>& inputSize)
{
  const int inputRows     = inputSize[0];
  const int inputCols     = inputSize[1];
  const int inputChannels = inputSize[2];

  const int filterRows     = filterSize_[0];
  const int filterCols     = filterSize_[1];
  // Filter and input channles must be equal
  const int filterChannels = inputSize[2];

  const int filterStrideRow = filterStride_[0];
  const int filterStrideCol = filterStride_[1];

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

  if (withPadding_)
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

  forwardConvDims_.setDimensions(
                                 inputRows, inputCols, inputChannels,
                                 inputObservationStride, inputChannelStride,
                                 padRowLeft, padRowRight,
                                 padColTop, padColBottom,
                                 inputRowsPadded, inputColsPadded,
                                 filterRows, filterCols, numberOfFilters_,
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
  backwardWeightsConvDims_.setDimensions(
                                         inputRows, inputCols, 0,
                                         inputObservationStrideBkw,
                                         inputChannelStrideBkw,
                                         padRowLeft, padRowRight,
                                         padColTop, padColBottom,
                                         inputRowsPadded, inputColsPadded,
                                         outputRows, outputCols, numberOfFilters_,
                                         filterStrideRow, filterStrideCol,
                                         filterRows, filterCols
                                        );


  //  ---------------------------------------------------------

  // -- Dimensions to perform the convolution of output with filter ---

  const int outputObservationStrideBkw = outputRows
                                       * outputCols
                                       * numberOfFilters_;

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
  backwardInputConvDims_.setDimensions(
                                       outputRows, outputCols, numberOfFilters_,
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


} // namespace
