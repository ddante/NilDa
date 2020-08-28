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
  const std::string mnistImagesTrainFile
    = "/home/dante/dev/NilDa/datasets/mnist/train-images-idx3-ubyte";

  const std::string mnistLabelsTrainFile
    = "/home/dante/dev/NilDa/datasets/mnist/train-labels-idx1-ubyte";

	NilDa::Matrix trainingImages;
	NilDa::Matrix trainingLabels;

	const NilDa::Scalar imgScaling = 1.0/255.0;

	NilDa::importMNISTDatabase(
                             mnistImagesTrainFile,
		                 			   mnistLabelsTrainFile,
		       			             imgScaling,
		       			             /*shuffle=*/ true,
			                       trainingImages,
			                       trainingLabels
                            );


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

	const int epochs = 7;
	const int batchSize = 32;

  nn.train(trainingImages, trainingLabels, epochs, batchSize);

  return 0;
}
