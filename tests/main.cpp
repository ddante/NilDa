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
    //trainingLabels << 1,0,0,1;

    NilDa::Matrix W1(2,3);
    W1 <<  -1, 2, -3, 0.4, -0.5, -0.6;
    NilDa::Vector b1(2);
    b1 << 0.3, 0.5;
    l1->setWeightsAndBiases(W1, b1);

    NilDa::Matrix W2(3,2);
    W2 << -1, 2, 0.3, 0.4, 0.6, 0.7;
    NilDa::Vector b2(3);
    b2 << 0.3, 0.5, 0.7;
    l2->setWeightsAndBiases(W2, b2);

    nn.setLossFunction("sparse_categorical_crossentropy");

    nn.forwardPropagation(trainingData);

    NilDa::Scalar J = nn.getLoss(trainingData, trainingLabels);
    std::cout << J << std::endl;

    nn.backwardPropagation(trainingData, trainingLabels);

    nn.gradientsSanity(trainingData, trainingLabels, /*printError=*/ true);

    return 0;
}