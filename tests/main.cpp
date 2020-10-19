#include <iostream>
#include <vector>
#include <string>

#include "utils/importDatasets.h"

#include "core/neuralNetwork/layers/inputLayer.h"
#include "core/neuralNetwork/layers/denseLayer.h"
#include "core/neuralNetwork/layers/conv2DLayer.h"
#include "core/neuralNetwork/layers/maxPool2DLayer.h"
#include "core/neuralNetwork/layers/dropoutLayer.h"
#include "core/neuralNetwork/layers/batchNormalizationLayer.h"
#include "core/neuralNetwork/neuralNetwork.h"
#include "core/neuralNetwork/optimizers/sgd.h"

int main(int argc, char const *argv[])
{
  NilDa::layer* l0 = new NilDa::inputLayer(3);
  NilDa::layer* l1 = new NilDa::denseLayer(4, "relu");
  NilDa::layer* l2 = new NilDa::batchNormalizationLayer();
  NilDa::layer* l3 = new NilDa::denseLayer(3, "softmax");

  NilDa::neuralNetwork nn({l0, l1, l2});

  nn.summary();

  NilDa::Matrix X(3, 5);
  X.setRandom();

  NilDa::Matrix Y(3, 5);
  Y << 1,0,0,0,1,
       0,1,0,1,0,
       0,0,1,0,0;

  const NilDa::Scalar learningRate = 0.01;
  const NilDa::Scalar momentum = 0.90;

  NilDa::sgd opt(learningRate, momentum);

  nn.configure(opt, "categorical_crossentropy");

  for (int i = 0; i < 50; ++i)
  {
    nn.forwardPropagation(X, true);
  }

  nn.forwardPropagation(X, false);

  //nn.backwardPropagation(X, Y);
  //nn.gradientsSanityCheck(X, Y, true);
}
