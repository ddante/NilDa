#include <iostream>
#include <vector>

#include "core/neuralNetwork/inputLayer.h"
#include "core/neuralNetwork/denseLayer.h"
#include "core/neuralNetwork/neuralNetwork.h"

int main(int argc, char const *argv[])
{
    NilDa::layer* l0 = new NilDa::inputLayer(10);
    NilDa::layer* l1 = new NilDa::denseLayer(15);
    NilDa::layer* l2 = new NilDa::denseLayer(2);

    NilDa::neuralNetwork nn({l0, l1, l2});

    //std::cout << l0->size() << std::endl;

    return 0;
}