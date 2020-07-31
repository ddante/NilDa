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
  const int ChI = 1;
  std::array<int, 3> inputSize = {rI, cI, ChI};

  const int rF = 2;
  const int cF = 2;
  std::array<int, 2> filterSize = {rF, cF};

  const int numberOfFilters = 1;

  std::array<int, 2> filterStride = {1, 1};

  bool padding = false;

  NilDa::layer* l0 = new NilDa::inputLayer(inputSize);

  NilDa::layer* l1 = new NilDa::conv2DLayer(
                                            numberOfFilters,
                                            filterSize,
                                            filterStride,
                                            padding,
                                            "relu"
                                           );

  NilDa::neuralNetwork nn({l0, l1});

  const int nObs = 1;
  NilDa::Matrix X(rI * cI, ChI * nObs);

  int l = 0;

  for (int i = 0; i < nObs; ++i)
  {
    for (int j = 0; j < ChI; ++j, ++l)
    {
      for (int k = 0; k < rI * cI; ++k)
      {
        X(k, l) = (k + 1) * (j + 1) * (i+1);
      }
    }
  }

  NilDa::errorCheck output = l1->localChecks(X, 1e-8);

  std::cout << output.error << std::endl;

  return 0;
}
