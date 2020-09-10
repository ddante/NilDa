#include <iostream>
#include <math.h>
#include "primitives/errors.h"
#include "core/neuralNetwork/layers/inputLayer.h"
#include "core/neuralNetwork/layers/denseLayer.h"
#include "core/neuralNetwork/layers/conv2DLayer.h"
#include "core/neuralNetwork/neuralNetwork.h"

const NilDa::Scalar errTol = 1e-10;

int runTest(
            const std::array<int, 3>& inputSize,
            const std::array<int, 2>& filterSize,
            const std::array<int, 2>& filterStride,
            const int nFilters,
            const bool padding,
            const int nObs
           )
{
  NilDa::layer* l0 = new NilDa::inputLayer(inputSize);

  NilDa::layer* l1 = new NilDa::conv2DLayer(
                                            nFilters,
                                            filterSize,
                                            filterStride,
                                            padding,
                                            "relu"
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

  // Test 1
  std::array<int, 3> inputSize{7, 7, 1};
  std::array<int, 2> filterSize{3, 3};
  std::array<int, 2> filterStride{1, 1};

  bool padding = false;

  int nFilters = 1;
  int nObs = 1;

  int test1 = runTest(
                      inputSize,
                      filterSize, filterStride, nFilters,
                      padding, nObs
                     );

  inputSize = {7, 7, 3};

  int test2 = runTest(
                      inputSize,
                      filterSize, filterStride, nFilters,
                      padding, nObs
                     );

  padding = true;

  int test3 = runTest(
                      inputSize,
                      filterSize, filterStride, nFilters,
                      padding, nObs
                     );

  nFilters = 5;

  int test4 = runTest(
                     inputSize,
                     filterSize, filterStride, nFilters,
                     padding, nObs
                    );

  filterSize = {2, 3};

  int test5 = runTest(
                     inputSize,
                     filterSize, filterStride, nFilters,
                     padding,nObs
                    );

  filterSize = {3, 2};

  int test6 = runTest(
                      inputSize,
                      filterSize, filterStride, nFilters,
                      padding,nObs
                     );

  inputSize = {5, 7, 4};

  int test7 = runTest(
                      inputSize,
                      filterSize, filterStride, nFilters,
                      padding,nObs
                     );

  inputSize = {7, 8, 4};

  int test8 = runTest(
                      inputSize,
                      filterSize, filterStride, nFilters,
                      padding,nObs
                     );

  inputSize = {7, 8, 3};
  filterSize = {3, 2};
  padding = true;
  nFilters = 5;
  nObs = 1;
  filterStride = {2, 2};

  int test9 = runTest(
                      inputSize,
                      filterSize, filterStride, nFilters,
                      padding,nObs
                     );

  filterStride = {2, 1};

  int test10 = runTest(
                       inputSize,
                       filterSize, filterStride, nFilters,
                       padding,nObs
                      );

  inputSize = {8, 7, 1};
  filterSize = {3, 2};
  padding = false;
  nFilters = 1;
  nObs = 1;
  filterStride = {2, 2};

  int test11 = runTest(
                       inputSize,
                       filterSize, filterStride, nFilters,
                       padding,nObs
                      );

  if (
      test1  == NilDa::EXIT_FAIL ||
      test2  == NilDa::EXIT_FAIL ||
      test3  == NilDa::EXIT_FAIL ||
      test4  == NilDa::EXIT_FAIL ||
      test5  == NilDa::EXIT_FAIL ||
      test6  == NilDa::EXIT_FAIL ||
      test7  == NilDa::EXIT_FAIL ||
      test8  == NilDa::EXIT_FAIL ||
      test9  == NilDa::EXIT_FAIL ||
      test10 == NilDa::EXIT_FAIL ||
      test11 == NilDa::EXIT_FAIL
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
