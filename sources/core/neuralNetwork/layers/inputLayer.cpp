#include "inputLayer.h"

// ---------------------------------------------------------------------------
namespace NilDa
{

inputLayer::inputLayer(const int inSize):
  inputSize_(inSize),
  inputRows_(0),
  inputCols_(0),
  inputChannels_(1),
  observationStride_(1),
  flattenLayer_(true)
{
  type_ = layerTypes::input;

  trainable_ = false;

  assert(inputSize_ > 0);
}

inputLayer::inputLayer(const std::array<int,3>& inSize):
  inputSize_(inSize[0]*inSize[1]*inSize[3]),
  inputRows_(inSize[0]),
  inputCols_(inSize[1]),
  inputChannels_(inSize[3]),
  observationStride_(inSize[3]),
  flattenLayer_(false)
{
  type_ = layerTypes::input;

  assert(inputRows_ >0);
  assert(inputCols_ > 0);
  assert(inputChannels_ >0);
}

void inputLayer::checkInputSize(const Matrix& obs) const
{
  if (flattenLayer_)
  {
    // For a flatten layer:
    // number of rows = number of features
    // number of cols = number of observations
    if (inputSize_ != obs.rows())
    {
      std::cerr << "Size of input data "
                << "(" << obs.rows() << ") "
                << " not consistent with input layer size"
                << "(" << inputSize_ << ") "
                << std::endl;

      assert(false);
    }

    numberOfObservations_ = obs.cols();
  }
  else
  {
    // For a 2D layer:
    // number of rows = number of features
    // number of cols = number of observations * number of channels
    const int channelSize = inputRows_*inputCols_;

    // TODO: how to check if the number of channels is correct?

    if (channelSize  !=obs.rows() ||
        obs.cols() % inputChannels_ != 0)
    {
      std::cerr << "Size of input data "
                << "(" << obs.rows() << ") "
                << " not consistent with input layer size"
                << "(" << channelSize << ") "
                << std::endl;
    }

    numberOfObservations_ = obs.cols() / inputChannels_;
  }
}


} // namespace
