#include "inputLayer.h"

// ---------------------------------------------------------------------------
namespace NilDa
{

inputLayer::inputLayer():
  observationStride_(0)
{
  type_ = layerTypes::input;

  size_.isFlat = true;
  size_.size = 0;
  size_.rows = 0;
  size_.cols = 0;
  size_.channels = 0;

  trainable_ = false;
}

inputLayer::inputLayer(const int inSize):
  observationStride_(1)
{
  type_ = layerTypes::input;

  size_.isFlat = true;
  size_.size = inSize;
  size_.rows = 0;
  size_.cols = 0;
  size_.channels = 0;

  trainable_ = false;
}

inputLayer::inputLayer(const std::array<int,3>& inSize):
  observationStride_(inSize[2])
{
  type_ = layerTypes::input;

  size_.isFlat = false;
  size_.size = inSize[0] * inSize[1] * inSize[2];
  size_.rows = inSize[0];
  size_.cols = inSize[1];
  size_.channels = inSize[2];

  trainable_ = false;
}

void inputLayer::checkInput() const
{
  assert(size_.size > 0);

  if (!size_.isFlat)
  {
    assert(size_.rows > 0);
    assert(size_.cols > 0);
    assert(size_.channels > 0);
  }
}

void inputLayer::checkInputSize(const Matrix& obs) const
{
  if (size_.isFlat)
  {
    // For a flatten layer:
    // number of rows = number of features
    // number of cols = number of observations
    if (size_.size != obs.rows())
    {
      std::cerr << "Size of input data "
                << "(" << obs.rows() << ") "
                << " not consistent with input layer size"
                << "(" << size_.size << ") "
                << std::endl;

      assert(false);
    }

  }
  else
  {
    // For a 2D layer:
    // number of rows = number of features
    // number of cols = number of observations * number of channels
    const int channelSize = size_.rows * size_.cols;

    // TODO: how to check if the number of channels is correct?

    if (channelSize != obs.rows() ||
        obs.cols() % size_.channels != 0)
    {
      std::cerr << "Size of input data "
                << "(" << obs.rows() << ") "
                << " not consistent with input layer size"
                << "(" << channelSize << ") "
                << std::endl;
    }
  }
}

void inputLayer::saveLayer(std::ofstream& ofs) const
{
  const int iType =
    static_cast<std::underlying_type_t<layerTypes> >(layerTypes::input);

  ofs.write((char*) (&iType), sizeof(int));

  ofs.write((char*) (&trainable_), sizeof(bool));

  ofs.write((char*) (&size_.isFlat), sizeof(bool));

  if (size_.isFlat)
  {
    ofs.write((char*) (&size_.size), sizeof(int));
  }
  else
  {
    ofs.write((char*) (&size_.rows),     sizeof(int));
    ofs.write((char*) (&size_.cols),     sizeof(int));
    ofs.write((char*) (&size_.channels), sizeof(int));
  }
}

void inputLayer::loadLayer(std::ifstream& ifs)
{
  ifs.read((char*) (&trainable_), sizeof(bool));

  ifs.read((char*) (&size_.isFlat), sizeof(bool));

  if (size_.isFlat)
  {
    ifs.read((char*) (&size_.size), sizeof(int));
    size_.rows = 0;
    size_.cols = 0;
  }
  else
  {
    ifs.read((char*) (&size_.rows),     sizeof(int));
    ifs.read((char*) (&size_.cols),     sizeof(int));
    ifs.read((char*) (&size_.channels), sizeof(int));

    size_.size = size_.rows * size_.cols * size_.channels;
  }
}


} // namespace
