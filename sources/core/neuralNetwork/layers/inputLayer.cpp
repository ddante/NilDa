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

void inputLayer::checkInputSize(const Matrix& inputData)
{
    if (flattenLayer_)
    {
        if (inputSize_ != inputData.rows())
        {
            std::cerr << "Size of the input data " 
                        << "(" << inputData.rows() << ") "
                        << " not consistent with the input layer size" 
                        << "(" << inputSize_ << ") "
                        << std::endl;

            assert(false);
        }
    }
    else
    {
        const int channelSize = inputRows_*inputCols_;

        if (channelSize  != inputData.rows())
        {
            std::cerr << "ISize of the input data " 
                        << "(" << inputData.rows() << ") "
                        << " not consistent with the input layer size" 
                        << "(" << channelSize << ") "
                        << std::endl;            
        }

        // TODO: how to check if the number of channels is correct?
    }
}


} // namespace
