#include <iostream>

#include "maxPool2DLayer.h"
#include "pool2DUtils.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

maxPool2DLayer::maxPool2DLayer(
                               const std::array<int, 2>& kernelSize,
  			                       const std::array<int, 2>& kernelStride,
  			                       const bool withPadding
                             ) :
  kernelSize_(kernelSize),
  kernelStride_(kernelStride),
  withPadding_(withPadding),
  poolDims_{}
{
  type_ = layerTypes::maxPool2D;

  trainable_ = false;
}

maxPool2DLayer::maxPool2DLayer(
                               const std::array<int, 2>& kernelSize,
  			                       const std::array<int, 2>& kernelStride
                              ) :
  kernelSize_(kernelSize),
  kernelStride_(kernelStride),
  withPadding_(false),
  poolDims_{}
{
  type_ = layerTypes::maxPool2D;

  trainable_ = false;
}

void maxPool2DLayer::init(const layer* previousLayer)
{
  if (previousLayer->layerType() != layerTypes::conv2D)
  {
    std::cerr << "Previous layer of type "
             <<  getLayerName(previousLayer->layerType())
             << " not compatible with current layer of type "
             << getLayerName(type_) << "." << std::endl;

    assert(false);
  }

  const layerSizes prevLayer = previousLayer->size();

  if (prevLayer.isFlat)
  {
    std::cerr << "Previous layer to " << getLayerName(type_)
              << " cannot be flat." << std::endl;
  }

  assert(prevLayer.rows > 0);
  assert(prevLayer.cols > 0);
  assert(prevLayer.channels > 0);

  const std::array<int, 3> prevLayerSize{
                                         prevLayer.rows,
                                         prevLayer.cols,
                                         prevLayer.channels
                                        };

  // This is not OK. Padding here means a different think
  setPool2DDims(
                prevLayerSize,
                kernelSize_,
                kernelStride_,
                withPadding_,
                poolDims_
               );

  size_.isFlat = false;

  size_.size = poolDims_.outputRows
             * poolDims_.outputCols
             * poolDims_.outputChannels;

  size_.rows = poolDims_.outputRows;
  size_.cols = poolDims_.outputCols;
  size_.channels = poolDims_.outputChannels;
}

void maxPool2DLayer::checkInputSize(const Matrix& inputData) const
{}

void maxPool2DLayer::forwardPropagation(const Matrix& input)
{}

void maxPool2DLayer::backwardPropagation(const Matrix& dActivationNext,
                                         const Matrix& input)
{}


} // namespace
