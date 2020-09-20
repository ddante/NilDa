#include <iostream>
#include <vector>
#include <math.h>

#include "primitives/errors.h"

#include "utils/importDatasets.h"

#include "core/neuralNetwork/layers/inputLayer.h"
#include "core/neuralNetwork/layers/denseLayer.h"
#include "core/neuralNetwork/layers/conv2DLayer.h"
#include "core/neuralNetwork/layers/maxPool2DLayer.h"

#include "core/neuralNetwork/neuralNetwork.h"
#include "core/neuralNetwork/optimizers/sgd.h"

int main(int argc, char const *argv[])
{
  const NilDa::Scalar errTol = 1e-10;

  NilDa::Scalar J, J2;

  const std::string mnistImagesTrainFile
    = "/home/dante/dev/NilDa/datasets/mnist/t10k-images-idx3-ubyte";

  const std::string mnistLabelsTrainFile
    = "/home/dante/dev/NilDa/datasets/mnist/t10k-labels-idx1-ubyte";

	NilDa::Matrix trainingImages;
	NilDa::Matrix trainingLabels;

	const NilDa::Scalar imgScaling = 1.0/255.0;

	NilDa::importMNISTDatabase(
                             mnistImagesTrainFile,
		                 			   mnistLabelsTrainFile,
		       			             imgScaling,
		       			             true,
                             /*sparseCategorical=*/ false,
			                       trainingImages,
			                       trainingLabels
                            );
  {
    NilDa::layer* l0 = new NilDa::inputLayer({28, 28, 1});

    NilDa::layer* l1 = new NilDa::conv2DLayer(32, {3,3}, true, "relu");

    NilDa::layer* l2 = new NilDa::maxPool2DLayer({2, 2}, {2, 2});

    NilDa::layer* l3 = new NilDa::conv2DLayer(64, {3,3}, true, "relu");

    NilDa::layer* l4 = new NilDa::maxPool2DLayer({2, 2}, {2, 2});

    NilDa::layer* l5 = new NilDa::denseLayer(128, "relu");

    NilDa::layer* l6 = new NilDa::denseLayer(10, "softmax");

    NilDa::neuralNetwork nn({l0, l1, l2, l3, l4, l5, l6});

    nn.summary();

    //

  	const NilDa::Scalar learningRate = 0.05;

  	const NilDa::Scalar momentum = 0.90;

    NilDa::sgd opt(learningRate, momentum);

    nn.configure(opt, "sparse_categorical_crossentropy");

    //

  	const int epochs = 1;
  	const int batchSize = 32;

    nn.train(trainingImages, trainingLabels, epochs, batchSize);

    J = nn.getAccuracy(trainingImages, trainingLabels);

    nn.saveModel("myConvNetModel.out");
  }

  {
    NilDa::neuralNetwork nn2;
    nn2.loadModel("myConvNetModel.out");

    nn2.summary();

    J2 = nn2.getAccuracy(trainingImages, trainingLabels);
  }

  NilDa::Scalar difference = fabs(J - J2);

  if (difference <  errTol)
  {
    std::cout << "test OK\n";
    return NilDa::EXIT_OK;
  }
  else
  {
    std::cout << "test FAILED, differences: "
              << difference << "\n";
    return NilDa::EXIT_FAIL;
  }
}
