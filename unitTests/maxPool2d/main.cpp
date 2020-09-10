#include <iostream>
#include <math.h>
#include "primitives/errors.h"
#include "core/neuralNetwork/layers/inputLayer.h"
#include "core/neuralNetwork/layers/denseLayer.h"
#include "core/neuralNetwork/layers/conv2DLayer.h"
#include "core/neuralNetwork/layers/maxPool2DLayer.h"
#include "core/neuralNetwork/neuralNetwork.h"

const NilDa::Scalar errTol = 1e-10;

int runTest(
            const std::array<int, 3>& inputSize,
            const std::array<int, 2>& kernelSize,
            const std::array<int, 2>& kernelStride,
            const int nObs
           )
{
  NilDa::layer* l0 = new NilDa::inputLayer(inputSize);

  NilDa::layer* l1 = new NilDa::maxPool2DLayer(
                                               kernelSize,
                                               kernelStride
                                              );

  NilDa::neuralNetwork nn({l0, l1});

  const int rowsI = inputSize[0]*inputSize[1];
  const int colsI = inputSize[2]*nObs;

  NilDa::Matrix X(rowsI, colsI);

  X.setRandom(rowsI, colsI);
/*
  int l = 0;
  for (int i = 0; i < nObs; ++i)
  {
    for (int j = 0; j < inputSize[2]; ++j, ++l)
    {
      for (int k = 0; k < rowsI; ++k)
      {
        X(k, l) = (k + 1) * (j + 1) * (i+1);
      }
    }
  }
*/

  NilDa::errorCheck output = l1->localChecks(X, 1e-8);

  std::cout << "Difference: " << output.error << " ";

  (output.code == NilDa::EXIT_OK) ?
                                  std::cout << "test OK\n"
                                  :
                                  std::cout << "test failed\n";

  return output.code;
}

int main(int argc, char const *argv[])
{
#ifdef ND_SP
    #warning "Single precision used. For testing specify either double or long precision"
#endif

  std::array<int, 3> inputSize{6, 6, 1};
  std::array<int, 2> kernelSize{2, 2};
  std::array<int, 2> kernelStride{1, 1};
  int nObs = 1;

  int test1 = runTest(
                      inputSize,
                      kernelSize, kernelStride, nObs
                     );

  inputSize = {7, 7, 3};
  kernelSize = {2, 2};
  kernelStride = {2, 2};
  nObs = 1;

  int test2 = runTest(
                      inputSize,
                      kernelSize, kernelStride, nObs
                     );

  inputSize = {7, 7, 3};
  kernelSize = {2, 2};
  kernelStride = {2, 2};
  nObs = 4;

  int test3 = runTest(
                      inputSize,
                      kernelSize, kernelStride, nObs
                     );

  inputSize = {7, 7, 3};
  kernelSize = {2, 2};
  kernelStride = {1, 1};
  nObs = 4;

  int test4 = runTest(
                      inputSize,
                      kernelSize, kernelStride, nObs
                     );

  inputSize = {7, 7, 3};
  kernelSize = {3, 3};
  kernelStride = {1, 1};
  nObs = 4;

  int test5 = runTest(
                      inputSize,
                      kernelSize, kernelStride,nObs
                     );

  inputSize = {6, 7, 3};
  kernelSize = {2, 3};
  kernelStride = {2, 1};
  nObs = 4;

  int test6 = runTest(
                      inputSize,
                      kernelSize, kernelStride,nObs
                     );

  inputSize = {7, 6, 3};
  kernelSize = {3, 2};
  kernelStride = {1, 2};
  nObs = 4;

  int test7 = runTest(
                     inputSize,
                     kernelSize, kernelStride,nObs
                    );

  if (
      test1  == NilDa::EXIT_FAIL ||
      test2  == NilDa::EXIT_FAIL ||
      test3  == NilDa::EXIT_FAIL ||
      test4  == NilDa::EXIT_FAIL ||
      test5  == NilDa::EXIT_FAIL ||
      test6  == NilDa::EXIT_FAIL ||
      test7  == NilDa::EXIT_FAIL
     )
  {
    return NilDa::EXIT_FAIL;
  }
  else
  {
    return NilDa::EXIT_OK;
  }


  return 0;
}
