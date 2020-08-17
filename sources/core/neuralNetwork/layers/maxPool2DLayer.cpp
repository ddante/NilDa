#include <iostream>

#include "maxPool2DLayer.h"
#include "pool2DUtils.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

maxPool2DLayer::maxPool2DLayer(
                               const std::array<int, 2>& kernelSize,
  			                       const std::array<int, 2>& kernelStride
                              ) :
  kernelSize_(kernelSize),
  kernelStride_(kernelStride),
  poolDims_{}
{
  type_ = layerTypes::maxPool2D;

  trainable_ = false;
}

void maxPool2DLayer::init(const layer* previousLayer)
{
  if (
      previousLayer->layerType() != layerTypes::input &&
      previousLayer->layerType() != layerTypes::conv2D
     )
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

void maxPool2DLayer::setupBackward(const layer* nextLayer)
{
  const layerSizes sLayer = nextLayer->size();

  undoFlattening_ = (sLayer.isFlat) ? true : false;
}

void maxPool2DLayer::checkInputSize(const Matrix& input) const
{
  const int inputSize = poolDims_.inputRows
                      * poolDims_.inputCols;

  if (input.rows() != inputSize)
  {
    std::cerr << "Size of input data "
    << "(" << input.rows() << ") "
    << " not consistent with maxPool2D layer size"
    << "(" << poolDims_.inputRows << ", "
    << poolDims_.inputCols << ") "
    << std::endl;

    assert(false);
  }

}

void maxPool2DLayer::checkInputAndCacheSize(
                                            const Matrix& input,
                                            const Matrix& cacheBackProp
                                          ) const
{
  checkInputSize(input);

  if (cacheBackProp.rows() != linearOutput_.rows() &&
      cacheBackProp.cols() != linearOutput_.cols() )
  {
    std::cerr << "Size of the back propagation cache "
    << "(" << cacheBackProp.rows() << ", "
           << cacheBackProp.cols() << ") "
    << " not consistent with the ouput size "
    << "(" << linearOutput_.rows() << ", "
           << linearOutput_.cols() << ") "
    << std::endl;

    assert(false);
  }
}

void maxPool2DLayer::forwardPropagation(const Matrix& input)
{
#ifdef ND_DEBUG_CHECKS
  checkInputSize(input);
#endif

  maxPool2D(poolDims_, input, linearOutput_, maxIndices_);
}

void maxPool2DLayer::backwardPropagation(const Matrix& dActivationNext,
                                         const Matrix& input)
{
  Matrix dLinearOutput(
                       linearOutput_.rows(),
                       linearOutput_.cols()
                      );

  if (undoFlattening_)
  {
    // The next layer is a flatten layer (dense)
    // so the cache must be rearranged in a 2D form
    ConstMapMatrix dActivationNextM(
                                    dActivationNext.data(),
                                    linearOutput_.rows(),
                                    linearOutput_.cols()
                                   );
#ifdef ND_DEBUG_CHECKS
  checkInputAndCacheSize(input, dActivationNextM);
#endif

    dLinearOutput = dActivationNextM.array();
  }
  else
  {
#ifdef ND_DEBUG_CHECKS
  checkInputAndCacheSize(input, dActivationNext);
#endif

    // The next layer is a 2d layer
    // no need to rearrange the cache
    dLinearOutput = dActivationNext.array();
  }

  const int cacheRows = poolDims_.inputRows
                      * poolDims_.inputCols;

  cacheBackProp_.setZero(
                         cacheRows,
                         dLinearOutput.cols()
                        );

  const Scalar* dout = dLinearOutput.data();

  const int* id = maxIndices_.data();

  Scalar* cache = cacheBackProp_.data();

  for (int i = 0; i < maxIndices_.size(); ++i)
  {
    cache[id[i]] += dout[i];
  }
}


} // namespace
