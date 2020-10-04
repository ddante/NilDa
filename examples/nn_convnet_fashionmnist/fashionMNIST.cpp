#include <iostream>
#include <vector>
#include <string>

#include "utils/importDatasets.h"
#include "utils/images.h"

#include "core/neuralNetwork/layers/inputLayer.h"
#include "core/neuralNetwork/layers/denseLayer.h"
#include "core/neuralNetwork/layers/conv2DLayer.h"
#include "core/neuralNetwork/layers/maxPool2DLayer.h"

#include "core/neuralNetwork/neuralNetwork.h"
#include "core/neuralNetwork/optimizers/sgd.h"
//#include "core/neuralNetwork/optimizers/adaGrad.h"
//#include "core/neuralNetwork/optimizers/rmsProp.h"
//#include "core/neuralNetwork/optimizers/adam.h"

int main(int argc, char const *argv[])
{
  const NilDa::Scalar imgScaling = 1.0/255.0;

   // Training
  const std::string mnistImagesTrainFile
    = "/home/dante/dev/NilDa/datasets/fashion_mnist/train-images-idx3-ubyte";

  const std::string mnistLabelsTrainFile
    = "/home/dante/dev/NilDa/datasets/fashion_mnist/train-labels-idx1-ubyte";

	NilDa::Matrix trainingImages;
	NilDa::Matrix trainingLabels;

 const bool shuffle = true;

  const bool sparseCategorical = true;

	NilDa::importMNISTDatabase(
                             mnistImagesTrainFile,
		                 			   mnistLabelsTrainFile,
		       			             imgScaling,
		       			             shuffle,
                             sparseCategorical,
			                       trainingImages,
			                       trainingLabels
                            );

  NilDa::layer* l0 = new NilDa::inputLayer({28, 28, 1});

  NilDa::layer* l1 = new NilDa::conv2DLayer(32, {3,3}, true, "relu");

  NilDa::layer* l2 = new NilDa::maxPool2DLayer({2, 2}, {2, 2});

  NilDa::layer* l3 = new NilDa::conv2DLayer(64, {3,3}, true, "relu");

  NilDa::layer* l4 = new NilDa::maxPool2DLayer({2, 2}, {2, 2});

  NilDa::layer* l5 = new NilDa::denseLayer(128, "sigmoid");

  NilDa::layer* l6 = new NilDa::denseLayer(10, "softmax");

  NilDa::neuralNetwork nn({l0, l1, l2, l3, l4, l5, l6});

  nn.summary();


  //

	const NilDa::Scalar learningRate = 0.001;

	const NilDa::Scalar momentum = 0.90;

  const NilDa::Scalar decay = 0.99;

  NilDa::sgd opt(learningRate, momentum);
  //NilDa::adaGrad opt(learningRate);
  //NilDa::rsmProp opt(learningRate, decay);
  //NilDa::adam opt(learningRate, decay, decay);

  nn.configure(opt, "sparse_categorical_crossentropy");

  //

	const int epochs = 20;
	const int batchSize = 32;

  nn.train(trainingImages, trainingLabels, epochs, batchSize);

  nn.saveModel("convNetFashionMnist.out");


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
		       			             shuffle,
                             sparseCategorical,
			                       predictImages,
			                       predictLabels
                            );

    std::cout << "Prediction Accuracy: "
              << nn.getAccuracy(predictImages, predictLabels) << "\n";

	/*
  NilDa::neuralNetwork nnT;

  nnT.loadModel("convNetFashionMnist.out");

  nnT.summary();

  std::cout << "Prediction Accuracy: "
            << nnT.getAccuracy(predictImages, predictLabels) << "\n";

  std::vector<std::string> classes
  {
    "T-shirt",
    "Trouses",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneakers",
    "Bag",
    "Boots"
  };

  for(int id = 0; id <  predictImages.cols(); ++id)
  {
    //NilDa::Matrix::Index trueLabel;
    //NilDa::Scalar max = predictLabels.col(id).maxCoeff(&trueLabel);

    NilDa::Matrix prob;
    nnT.getProbability(predictImages.col(id), prob);

    NilDa::Matrix::Index prediction;
    NilDa::Scalar max2 = prob.col(0).maxCoeff(&prediction);

    int ic = static_cast<int>(prediction);

    NilDa::displayImage(predictImages, {28,28,1}, id, classes[ic]);
  }
	*/
}
