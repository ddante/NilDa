#include "inputLayer.h"

// --------------------------------------------------------------------------- 
namespace NilDa
{

inputLayer::inputLayer(const int inSize):
    inputSize_(inSize),
    flattenLayer_(true)
{
    type_ = layerTypes::input;

    assert(inputSize_ > 0);     
}

inputLayer::inputLayer(const std::array<int,3>& inSize):
    inputRows_(inSize[0]),
    inputCols_(inSize[1]),
    inputChannels_(inSize[3]),
    flattenLayer_(false)
{    
    type_ = layerTypes::input;

    assert(inputRows_ >0);
    assert(inputCols_ > 0);
    assert(inputChannels_ >0);
}


} // namespace
