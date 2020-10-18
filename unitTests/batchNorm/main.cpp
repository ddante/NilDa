#include <iostream>
#include <math.h>
#include "primitives/errors.h"
#include "core/neuralNetwork/layers/inputLayer.h"
#include "core/neuralNetwork/layers/denseLayer.h"
#include "core/neuralNetwork/layers/batchNormalizationLayer.h"
#include "core/neuralNetwork/neuralNetwork.h"

const NilDa::Scalar errTol = 1e-10;

int test1()
{
  NilDa::layer* l0 = new NilDa::inputLayer(3);
  NilDa::layer* l1 = new NilDa::denseLayer(4, "relu");
  NilDa::layer* l2 = new NilDa::batchNormalizationLayer();

  NilDa::neuralNetwork nn({l0, l1, l2});

  NilDa::Matrix trainingData(3, 5);
  trainingData <<  1.1, 1.21, 0.2, 0.42,  0.9
                   0.3, 0.63, 4.4, 4.84, -0.1
                   5.1, -0.3, 0.9, -1.4,  1.13;

    NilDa::Matrix trainingLabels(3, 5);
    trainingLabels << 1,0,0,0,1
                      0,1,0,1,0
                      0,0,1,0,0;

    nn.setLossFunction("categorical_crossentropy");

    nn.forwardPropagation(trainingData);

    //nn.backwardPropagation(trainingData, trainingLabels);
}

int main(int argc, char const *argv[])
{
#ifdef ND_SP
    #warning "Single precision used. For testing specify either double or long precision"
#else

    if (test1() == NilDa::EXIT_OK)
    {
        return NilDa::EXIT_OK;
    }
    else
    {
        return NilDa::EXIT_FAIL;
    }

#endif
}
