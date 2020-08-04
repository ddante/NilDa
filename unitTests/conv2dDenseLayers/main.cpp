#include <iostream>
#include <math.h>
#include "primitives/errors.h"
#include "core/neuralNetwork/layers/inputLayer.h"
#include "core/neuralNetwork/layers/denseLayer.h"
#include "core/neuralNetwork/layers/conv2DLayer.h"
#include "core/neuralNetwork/neuralNetwork.h"

const NilDa::Scalar errTol = 1e-10;

int main(int argc, char const *argv[])
{
  const int n = 5;

  const int rI = n;
  const int cI = n;
  const int chI = 5;

  const int rF = 3;
  const int cF = 3;

  const int rS = 1;
  const int cS = 1;

  const int nFilters = 2;

  const bool padding = true;

  NilDa::layer* l0 = new NilDa::inputLayer({rI, cI, chI});

  NilDa::layer* l1 = new NilDa::conv2DLayer(
                                            nFilters,
                                            {rF, cF},
                                            {rS, cS},
                                            padding,
                                            "relu"
                                           );
  const int nNodes = 10;

  NilDa::layer* l2 = new NilDa::denseLayer(nNodes, "softmax");

  NilDa::neuralNetwork nn({l0, l1, l2});

  const int nObs = 1;
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

  NilDa::Matrix W(rF*cF, nFilters);
  W.setOnes(rF*cF, chI*nFilters);
  W *= 0.002;

  NilDa::Vector b(nFilters);
  b.setOnes(nFilters);

  l1->setWeightsAndBiases(W, b);

  NilDa::Matrix W1(nNodes, rI*cI*nFilters);
  W1.setOnes(nNodes, rI*cI*nFilters);
  W1 *= 0.001;

  NilDa::Vector b1(nNodes);
  b1.setOnes(nNodes);

  l2->setWeightsAndBiases(W1, b1);

  nn.forwardPropagation(X);

  std::cout << l2->output() << "\n\n";


/*
  if()
  {
    return NilDa::EXIT_FAIL;
  }
  else
  {
    return NilDa::EXIT_OK;
  }
*/

  return 0;
}
