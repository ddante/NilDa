#include <iostream>
#include "primitives/errors.h"
#include "core/neuralNetwork/layers/activationFunctions/activationFunction.h"
#include "core/neuralNetwork/layers/activationFunctions/identity.h"
#include "core/neuralNetwork/layers/activationFunctions/relu.h"
#include "core/neuralNetwork/layers/activationFunctions/sigmoid.h"
#include "core/neuralNetwork/layers/activationFunctions/softmax.h"

const NilDa::Scalar errTol = 1e-10;

int checkIdentity(const NilDa::Matrix& input, const NilDa::Matrix& G)
{
    NilDa::activationFunction* activation= new  NilDa::identity;

    NilDa::Matrix output;
    output.resize(input.rows(), input.cols());
    activation->applyBackward(input, G, output);

    NilDa::Matrix checkData;
    checkData.resize(3,4);
    checkData << -0.47, 0.66, 0.28, 0.87,
                       -0.21, -0.54, 0.11, 0.13,
                        0.97, 0.04,  0.00, -0.35;

    NilDa::Scalar difference = (output - checkData).norm();

    std::cout << " Identity activation function: ";
    if (difference < errTol)
    {
         std::cout << "test OK\n";
         return NilDa::EXIT_OK;
    }
    else
    {
        std::cout << "test FAILED, difference = " <<difference << "\n";
        return NilDa::EXIT_FAIL;
    }
}

int checkRelu(const NilDa::Matrix& input, const NilDa::Matrix& G)
{
    NilDa::activationFunction* activation = new  NilDa::relu;

    NilDa::Matrix output;
    output.resize(input.rows(), input.cols());
    activation->applyBackward(input, G, output);

    NilDa::Matrix checkData;
    checkData.resize(3,4);
    checkData << 0, 0.66, 0.28, 0,
                      0, -0.54, 0.11, 0.13,
                      0, 0.04,  0.0, 0.0;

    NilDa::Scalar difference = (output - checkData).norm();

    std::cout << " ReLU     activation function: ";
    if (difference < errTol)
    {
         std::cout << "test OK\n";
         return NilDa::EXIT_OK;
    }
    else
    {
        std::cout << "test FAILED, difference = " << difference << "\n";
        return NilDa::EXIT_FAIL;
    }
}

int checkSigmoid(const NilDa::Matrix& input, const NilDa::Matrix& G)
{
    NilDa::activationFunction* activation = new  NilDa::sigmoid;

    NilDa::Matrix output;
    output.resize(input.rows(), input.cols());
    activation->applyBackward(input, G, output);

    NilDa::Matrix checkData;
    checkData.resize(3,4);
    checkData << -0.100154151027753, 0.154719406994302, 0.0614440726696785, 0.213583731629868,
                      -0.0467155456371567, -0.109554293031198, 0.0268497522282723, 0.0311078582096665,
                       0.237808423053721,   0.00999900006666289, 0,   -0.0874803154527485;

    NilDa::Scalar difference = (output - checkData).norm();

    std::cout << " Sigmoid  activation function: ";
    if (difference < errTol)
    {
         std::cout << "test OK\n";
         return NilDa::EXIT_OK;
    }
    else
    {
        std::cout << "test FAILED, difference = " << difference << "\n";
        return NilDa::EXIT_FAIL;
    }
}

int checkSoftMax(const NilDa::Matrix& input, const NilDa::Matrix& G)
{
    NilDa::activationFunction* activation = new  NilDa::softmax;

    NilDa::Matrix output;
    output.resize(input.rows(), input.cols());
    activation->applyBackward(input, G, output);

    NilDa::Matrix checkData;
    checkData.resize(3,4);
    checkData << -0.187112007445907, 0.224533146798003, 0.0544004011009916, 0.166370870996672,
                       -0.1343565153556, -0.240914779413685, -0.0152684415430404, -0.0142268374992013,
                        0.321468522801508, 0.0163816326156819, -0.0391319595579512, -0.152144033497471;

    NilDa::Scalar difference = (output - checkData).norm();

    std::cout << " SoftMax  activation function: ";
    if (difference < errTol)
    {
         std::cout << "test OK\n";
         return NilDa::EXIT_OK;
    }
    else
    {
        std::cout << "test FAILED, difference = " << difference << "\n";
        return NilDa::EXIT_FAIL;
    }
}


int main(int argc, char const *argv[])
{
#ifdef ND_SP
    #warning"Single precision used. For testing specify either double or long precision"
#else

    NilDa::Matrix inputData;
    inputData.resize(3,4);
    inputData << -0.81, 0.51,  0.73, -0.27,
                      -0.69,  0.93, 0.31,  0.42,
                      -0.28,  0.02, 0.10, -0.03;

    NilDa::Matrix G;
    G.resize(3,4);
    G << -0.47, 0.66, 0.28, 0.87,
            -0.21, -0.54, 0.11, 0.13,
             0.97, 0.04,  0.00, -0.35;

    if (checkIdentity(inputData, G) == NilDa::EXIT_OK &&
        checkRelu(inputData, G)     == NilDa::EXIT_OK &&
        checkSigmoid(inputData, G) == NilDa::EXIT_OK &&
        checkSoftMax(inputData, G) == NilDa::EXIT_OK)
    {
        return NilDa::EXIT_OK;
    }
    else
    {
        return NilDa::EXIT_FAIL;
    }

#endif
}
