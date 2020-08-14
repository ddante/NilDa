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
  const int n = 3;

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

  NilDa::layer* l1 = new NilDa::maxPool2DLayer(
                                               {2, 2},
                                               {2, 2},
                                               true
                                              );

  NilDa::neuralNetwork nn({l0, l1});

  nn.summary();

  const int nObs = 1;

  NilDa::Matrix X(rI * cI, chI * nObs);

  int l = 0;
  for (int i = 0; i < nObs; ++i)
  {
    for (int j = 0; j < chI; ++j, ++l)
    {
        for (int k = 0; k < rI * cI; ++k)
        {
            X(k, l) = (k + 1) * (j + 1) * (i+1) - 1;
        }
    }
  }
  std::cout << X << "\n---------\n";

  //X.setRandom(rI * cI, chI * nObs);

  nn.forwardPropagation(X);
//nn.backwardPropagation(X,Y);

//  int out = nn.gradientsSanityCheck(X, Y, true);

  return 0;
}
