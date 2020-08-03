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

  NilDa::layer* l2 = new NilDa::denseLayer(10, "softmax");

  NilDa::neuralNetwork nn({l0, l1, l2});

  const int nObs = 3;
  NilDa::Matrix X(rI * cI, chI * nObs);

  int l = 0;

  for (int i = 0; i < nObs; ++i)
  {
    for (int j = 0; j < chI; ++j, ++l)
    {
      for (int k = 0; k < rI * cI; ++k)
      {
        X(k, l) = (k + 1) * (j + 1) * (i+1);
      }
    }
  }


  nn.forwardPropagation(X);

  return 0;
}
