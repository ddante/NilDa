#include <iostream>
#include <assert.h>
#include <memory>

#include "layer.h"
#include "inputLayer.h"
#include "denseLayer.h"
#include "conv2DLayer.h"
#include "maxPool2DLayer.h"
#include "dropoutLayer.h"
#include "batchNormalizationLayer.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

std::string getLayerName(const layerTypes inLayerType)
{
  std::string name;

  if (inLayerType == layerTypes::input)
  {
    name = "Input";
  }
  else if (inLayerType == layerTypes::dense)
  {
    name = "Dense";
  }
  else if (inLayerType == layerTypes::conv2D)
  {
    name = "Conv 2D";
  }
  else if (inLayerType == layerTypes::maxPool2D)
  {
    name = "MaxPool 2D";
  }
  else if (inLayerType == layerTypes::dropout)
  {
    name = "Dropout";
  }
  else if (inLayerType == layerTypes::batchNormalization)
  {
    name = "Batch normalization";
  }
  else
  {
    std::cerr << "Unknown layer type code.\n";
    assert(false);
  }

  return name;
}

layer* createLayer(const int layerCode)
{
  layerTypes type = static_cast<layerTypes>(layerCode);

  layer* newLayer;

  if (type == layerTypes::input)
  {
    newLayer = new inputLayer();
  }
  else if (type == layerTypes::dense)
  {
    newLayer = new denseLayer();
  }
  else if (type == layerTypes::conv2D)
  {
    newLayer = new conv2DLayer();
  }
  else if (type == layerTypes::maxPool2D)
  {
    newLayer = new maxPool2DLayer();
  }
  else if (type == layerTypes::dropout)
  {
    newLayer = new dropoutLayer();
  }
  else if (type == layerTypes::batchNormalization)
  {
    newLayer = new batchNormalizationLayer();
  }
  else
  {
    std::cerr << "Unknown layer type code.\n";
    assert(false);
  }

  return newLayer;
}

} // namespace
