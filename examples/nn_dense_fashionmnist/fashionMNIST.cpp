#include <iostream>
#include <vector>

#include "utils/importDatasets.h"
#include "core/neuralNetwork/layers/inputLayer.h"
#include "core/neuralNetwork/layers/denseLayer.h"
#include "core/neuralNetwork/neuralNetwork.h"
#include "core/neuralNetwork/optimizers/sgd.h"
#include "core/neuralNetwork/optimizers/adaGrad.h"
#include "core/neuralNetwork/optimizers/rsmProp.h"

int main(int argc, char const *argv[])
{
   // Training
  const std::string mnistImagesTrainFile
    = "/home/dante/dev/NilDa/datasets/fashion_mnist/train-images-idx3-ubyte";

  const std::string mnistLabelsTrainFile
    = "/home/dante/dev/NilDa/datasets/fashion_mnist/train-labels-idx1-ubyte";

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
  NilDa::layer* l1 = new NilDa::denseLayer(28, "sigmoid");
  NilDa::layer* l2 = new NilDa::denseLayer(10, "softmax");

  NilDa::neuralNetwork nn({l0, l1, l2});

  //

	const NilDa::Scalar learningRate = 0.1;

	const NilDa::Scalar momentum = 0.90;

  const NilDa::Scalar decay = 0.9;

  NilDa::sgd opt(learningRate, momentum);
  //NilDa::adaGrad opt(learningRate);
  //NilDa::rsmProp opt(learningRate, decay);

  nn.configure(opt, "sparse_categorical_crossentropy");

  //

	const int epochs = 20;
	const int batchSize = 32;

  nn.train(trainingImages, trainingLabels, epochs, batchSize);

  const std::string mnistImagesPredictFile
    = "/home/dante/dev/NilDa/datasets/fashion_mnist/t10k-images-idx3-ubyte";

  const std::string mnistLabelsPredictFile
    = "/home/dante/dev/NilDa/datasets/fashion_mnist/t10k-labels-idx1-ubyte";

	NilDa::Matrix predictImages;
	NilDa::Matrix predictLabels;

	NilDa::importMNISTDatabase(
                             mnistImagesPredictFile,
		                 			   mnistLabelsPredictFile,
		       			             imgScaling,
		       			             /*shuffle=*/ true,
			                       predictImages,
			                       predictLabels
                            );

    std::cout << "Prediction Accuracy: "
              << nn.getAccuracy(predictImages, predictLabels) << "\n";

}
