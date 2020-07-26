#include <iostream>
#include <vector>

#include "utils/importDatasets.h"
#include "core/neuralNetwork/layers/inputLayer.h"
#include "core/neuralNetwork/layers/denseLayer.h"
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

  NilDa::layer* l0 = new NilDa::inputLayer(784);
  NilDa::layer* l1 = new NilDa::denseLayer(28, "relu");
  NilDa::layer* l2 = new NilDa::denseLayer(10, "softmax");

  NilDa::neuralNetwork nn({l0, l1, l2});

  //

	const NilDa::Scalar learningRate = 0.1;

	const NilDa::Scalar momentum = 0.90;

  NilDa::sgd opt(learningRate, momentum);

  nn.configure(opt, "sparse_categorical_crossentropy");

  //

	const int epochs = 10;
	const int batchSize = 32;

  nn.train(trainingImages, trainingLabels, epochs, batchSize);

  return 0;
}
