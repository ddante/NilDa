#include <iostream>
#include <vector>

#include "utils/importDatasets.h"
#include "utils/images.h"

#include "core/neuralNetwork/layers/inputLayer.h"
#include "core/neuralNetwork/layers/denseLayer.h"
#include "core/neuralNetwork/neuralNetwork.h"
#include "core/neuralNetwork/optimizers/sgd.h"

int main(int argc, char const *argv[])
{
   // Training
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
  		       			             true,
  			                       trainingImages,
  			                       trainingLabels
                              );

    NilDa::layer* l0 = new NilDa::inputLayer(784);
    NilDa::layer* l1 = new NilDa::denseLayer(28, "sigmoid");
    NilDa::layer* l2 = new NilDa::denseLayer(10, "softmax");

    NilDa::neuralNetwork nn({l0, l1, l2});

    nn.summary();

    //

  	const NilDa::Scalar learningRate = 0.1;

  	const NilDa::Scalar momentum = 0.90;

    NilDa::sgd opt(learningRate, momentum);

    nn.configure(opt, "sparse_categorical_crossentropy");

    //

  	const int epochs = 10;
  	const int batchSize = 32;

    nn.train(trainingImages, trainingLabels, epochs, batchSize, 2);

    // Save the trained model
    nn.saveModel("myModel.out");
  }

  // prediciton
  {
    const std::string mnistImagesPredictFile
      = "/home/dante/dev/NilDa/datasets/mnist/t10k-images-idx3-ubyte";

    const std::string mnistLabelsPredictFile
      = "/home/dante/dev/NilDa/datasets/mnist/t10k-labels-idx1-ubyte";

  	NilDa::Matrix predictImages;
  	NilDa::Matrix predictLabels;

  	const NilDa::Scalar imgScaling = 1.0/255.0;

  	NilDa::importMNISTDatabase(
                               mnistImagesPredictFile,
  		                 			   mnistLabelsPredictFile,
  		       			             imgScaling,
  		       			             true,
  			                       predictImages,
  			                       predictLabels
                              );

    NilDa::neuralNetwork nnT;

    nnT.loadModel("myModel.out");

    nnT.summary();

    std::cout << "Prediction Accuracy: "
              << nnT.getAccuracy(predictImages, predictLabels) << "\n";

    int totErr = 0;

    for(int id = 0; id <  predictImages.cols(); ++id)
    {
      NilDa::Matrix::Index trueLabel;
      NilDa::Scalar max = predictLabels.col(id).maxCoeff(&trueLabel);

      NilDa::Matrix prob;
      nnT.getProbability(predictImages.col(id), prob);

      NilDa::Matrix::Index prediction;
      NilDa::Scalar max2 = prob.col(0).maxCoeff(&prediction);

      //NilDa::displayImage(predictImages, {28,28,1}, id, std::to_string(prediction));

      //std::cout << trueLabel << ", " << prediction;
    }
  }

}
