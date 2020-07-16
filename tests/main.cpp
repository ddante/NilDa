#include <iostream>
#include <vector>

#include "core/neuralNetwork/layers/inputLayer.h"
#include "core/neuralNetwork/layers/denseLayer.h"
#include "core/neuralNetwork/neuralNetwork.h"

int main(int argc, char const *argv[])
{
    NilDa::layer* l0 = new NilDa::inputLayer(3);
    NilDa::layer* l1 = new NilDa::denseLayer(2, "relu");
    NilDa::layer* l2 = new NilDa::denseLayer(3, "softmax");

    NilDa::neuralNetwork nn({l0, l1, l2});

    NilDa::Matrix trainingData(3, 4);
    trainingData <<  1.1, 1.21, 0.2, 0.42, 0.3, 0.63, 4.4, 4.84, 5.1, -0.3, 0.9, -1.4;

    NilDa::Matrix trainingLabels(3, 4);
    trainingLabels << 1,0,0,0, 0,1,0,1 ,0,0,1,0;

    nn.setLossFunction("sparse_categorical_crossentropy");
    nn.forwardPropagation(trainingData);
    NilDa::Scalar J = nn.getLoss(trainingData, trainingLabels);
    std::cout << J << std::endl;

    return 0;
}