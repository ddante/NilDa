#include <iostream>

#include "core/neuralNetwork/layers/activationFunctions/activationFunction.h"
#include "core/neuralNetwork/layers/activationFunctions/identity.h"
#include "core/neuralNetwork/layers/activationFunctions/relu.h"
#include "core/neuralNetwork/layers/activationFunctions/sigmoid.h"
#include "core/neuralNetwork/layers/activationFunctions/softmax.h"

const NilDa::Scalar errTol = 1e-10;

int checkIdentity(const NilDa::Matrix& input)
{ 
    NilDa::activationFunction* activation= new  NilDa::identity;

    NilDa::Matrix output = activation->applyForward(input);

    NilDa::Matrix checkData;
    checkData.resize(3,4);
    checkData << -0.81, 0.51,  0.73, -0.27,
                       -0.69,  0.93, 0.31,  0.42,
                       -0.28,  0.02, 0.10, -0.03;

    NilDa::Scalar difference = (output - checkData).norm();

    std::cout << " Identity activation function: ";
    if (difference < errTol)
    {
         std::cout << "test OK\n";
         return 1;
    }
    else
    {
        std::cout << "test FAILED, difference = " <<difference << "\n";
        return 0;
    }
}

int checkRelu(const NilDa::Matrix& input)
{
    NilDa::activationFunction* activation = new  NilDa::relu;
   
    NilDa::Matrix output = activation->applyForward(input);

    NilDa::Matrix checkData;
    checkData.resize(3,4);
    checkData << 0, 0.51, 0.73, 0,
                       0, 0.93, 0.31, 0.42,
                       0, 0.02, 0.10, 0;

    NilDa::Scalar difference = (output - checkData).norm();
    
    std::cout << " ReLU     activation function: ";
    if (difference < errTol)
    {
         std::cout << "test OK\n"; 
         return 1;
    }
    else
    {
        std::cout << "test FAILED, difference = " << difference << "\n";
        return 0;
    }
}

int checkSigmoid(const NilDa::Matrix& input)
{
    NilDa::activationFunction* activation = new  NilDa::sigmoid;
   
    NilDa::Matrix output = activation->applyForward(input);

    NilDa::Matrix checkData;
    checkData.resize(3,4);
    checkData << 0.307890495698212, 0.624806474468429, 0.674805272582313, 0.432907095034546,
                      0.33403307324818,   0.717075285492973,   0.576885261132046, 0.603483249864726,
                      0.430453776060771,  0.50499983334,        0.52497918747894,   0.49250056244938;

    NilDa::Scalar difference = (output - checkData).norm();
    
    std::cout << " Sigmoid  activation function: ";
    if (difference < errTol)
    {
         std::cout << "test OK\n";
         return 1;
    }
    else
    {
        std::cout << "test FAILED, difference = " << difference << "\n";
        return 0;
    }
}
int main(int argc, char const *argv[])
{
    int testOK = 0;

#ifdef ND_SP

    #error "Single precision used. For testing specify either double or long precision"

#else

    NilDa::Matrix inputData;
    inputData.resize(3,4);
    inputData << -0.81, 0.51,  0.73, -0.27,
                      -0.69,  0.93, 0.31,  0.42,
                      -0.28,  0.02, 0.10, -0.03;

    int codeI = checkIdentity(inputData);

    int codeR = checkRelu(inputData);

    int codeS = checkSigmoid(inputData);

    if (codeI && codeR && codeS)
    {
        testOK = 1;
    }
   
#endif

    return testOK;
}