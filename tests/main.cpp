#include <iostream>
#include <vector>

#include "utils/importDatasets.h"

#include "core/neuralNetwork/layers/inputLayer.h"
#include "core/neuralNetwork/layers/denseLayer.h"
#include "core/neuralNetwork/layers/conv2DLayer.h"

#include "core/neuralNetwork/neuralNetwork.h"
#include "core/neuralNetwork/optimizers/sgd.h"

int main(int argc, char const *argv[])
{
  NilDa::layer* l0 = new NilDa::inputLayer({5,5,3});

  NilDa::layer* l1 = new NilDa::conv2DLayer(
                                            7,
                                            {3,3},
                                            {1,1},
                                            true,
                                            "relu"
                                           );

  NilDa::layer* l2 = new NilDa::denseLayer(10, "softmax");

  NilDa::neuralNetwork nn({l0, l1, l2});

  //

  return 0;
}
