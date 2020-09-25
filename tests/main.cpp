#include <iostream>
#include <vector>

#include "utils/importDatasets.h"

#include "core/neuralNetwork/layers/inputLayer.h"
#include "core/neuralNetwork/layers/denseLayer.h"
#include "core/neuralNetwork/layers/conv2DLayer.h"
#include "core/neuralNetwork/layers/maxPool2DLayer.h"
#include "core/neuralNetwork/layers/dropoutLayer.h"
#include "core/neuralNetwork/neuralNetwork.h"
#include "core/neuralNetwork/optimizers/sgd.h"

int main(int argc, char const *argv[])
{
  NilDa::layer* l0 = new NilDa::inputLayer(6);
  NilDa::layer* l1 = new NilDa::denseLayer(5, "sigmoid");
  NilDa::layer* l2 = new NilDa::dropoutLayer(0.9);
  NilDa::layer* l3 = new NilDa::denseLayer(3, "softmax");

  NilDa::neuralNetwork nn({l0, l1, l2, l3});

  NilDa::Matrix trainingData;
  trainingData.setRandom(6,4);

  NilDa::Matrix trainingLabels(3, 4);
  trainingLabels << 1,0,0,0,
                    0,1,0,1,
                    0,0,1,0;

  nn.setLossFunction("categorical_crossentropy");

  nn.forwardPropagation(trainingData);
  nn.backwardPropagation(trainingData, trainingLabels);

  nn.gradientsSanityCheck(trainingData, trainingLabels, true);

  return 0;
}
