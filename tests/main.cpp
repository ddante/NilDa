#include <iostream>
#include <vector>

//#include "core/neuralNetwork/coreNilDa.h"

#include "core/neuralNetwork/layers/inputLayer.h"
#include "core/neuralNetwork/layers/denseLayer.h"
#include "core/neuralNetwork/neuralNetwork.h"

int main(int argc, char const *argv[])
{
    NilDa::layer* l0 = new NilDa::inputLayer(10);
    NilDa::layer* l1 = new NilDa::denseLayer(15, "relu");
    NilDa::layer* l2 = new NilDa::denseLayer(15, "relu");
    NilDa::layer* l3 = new NilDa::denseLayer(15, "relu");
    NilDa::layer* l4 = new NilDa::denseLayer(15, "relu");
    NilDa::layer* l5 = new NilDa::denseLayer(2, "relu");

    NilDa::neuralNetwork nn({l0, l1, l2,l3,l4,l5});

    std::cout << l0->size() << std::endl;

    return 0;
}