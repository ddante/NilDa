#include <iostream>
#include <vector>

#include "utils/importDatasets.h"

#include "core/neuralNetwork/layers/inputLayer.h"
#include "core/neuralNetwork/layers/denseLayer.h"
#include "core/neuralNetwork/layers/conv2DLayer.h"
#include "core/neuralNetwork/layers/maxPool2DLayer.h"

#include "core/neuralNetwork/neuralNetwork.h"
#include "core/neuralNetwork/optimizers/sgd.h"

int main(int argc, char const *argv[])
{
  const int n = 5;

  const int rI = n;
  const int cI = n;
  const int chI = 1;

  const int rF = 2;
  const int cF = 2;

  const int rS = 1;
  const int cS = 1;

  const int nFilters = 3;

  const bool padding = true;

  NilDa::layer* l0 = new NilDa::inputLayer({rI, cI, chI});

  NilDa::layer* l1 = new NilDa::conv2DLayer(
                                            nFilters,
                                            {rF, cF},
                                            {rS, cS},
                                            padding,
                                            "relu"
                                           );

  NilDa::layer* l2 = new NilDa::maxPool2DLayer(
                                               {2, 2},
                                               {1, 1},
                                               true
                                              );

  NilDa::layer* l3 = new NilDa::conv2DLayer(
                                            2*nFilters,
                                            {rF, cF},
                                            {rS, cS},
                                            padding,
                                            "relu"
                                           );

  NilDa::layer* l4 = new NilDa::denseLayer(3, "softmax");

  NilDa::neuralNetwork nn({l0, l1, l2, l3, l4});

  nn.summary();

  nn.setLossFunction("sparse_categorical_crossentropy");

  const int nObs = 4;
  NilDa::Matrix X(rI * cI, chI * nObs);

  X.setRandom(rI * cI, chI * nObs);

  NilDa::Matrix Y(3, nObs);
  Y << 1,0,0,0,
       0,1,0,1,
       0,0,1,0;

//nn.forwardPropagation(X);
//nn.backwardPropagation(X,Y);

//  int out = nn.gradientsSanityCheck(X, Y, true);

  return 0;
}
