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
  const int n = 4;

  const int rI = n;
  const int cI = n;
  const int chI = 1;

  const int rF = 2;
  const int cF = 2;

  const int rS = 1;
  const int cS = 1;

  const int nFilters = 1;

  const bool padding = true;

  NilDa::layer* l0 = new NilDa::inputLayer({rI, cI, chI});

  NilDa::layer* l1 = new NilDa::maxPool2DLayer(
                                               {2, 2},
                                               {2, 2}
                                              );

  NilDa::neuralNetwork nn({l0, l1});

  nn.setLossFunction("sparse_categorical_crossentropy");

  nn.summary();

  const int nObs = 5;

  NilDa::Matrix X(rI * cI, chI * nObs);

  int l = 0;
  int p = 0;
  for (int i = 0; i < nObs; ++i)
  {
    for (int j = 0; j < chI; ++j, ++l)
    {
        for (int k = 0; k < rI * cI; ++k, ++p)
        {
            X(k, l) = (k + 1) * (j + 1) * (i+1) - 1;
        }
    }
  }

  X.setRandom(rI * cI, chI * nObs);

  std::cout << X << "\n---------\n";
/*
  NilDa::Matrix Y(3, nObs);
  Y << 1,0,0,0,
       0,1,0,1,
       0,0,1,0;
*/
  nn.forwardPropagation(X);
  l1->localChecks(X, 1e-8);

  //nn.backwardPropagation(X, Y);

  //std::cout << nn.gradientsSanityCheck(X, Y, true);

  return 0;
}
