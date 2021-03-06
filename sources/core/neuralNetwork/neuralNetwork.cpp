#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include "primitives/errors.h"

#include "utils/progressBar.h"
#include "utils/images.h"

#include "neuralNetwork.h"

#include "lossFunctions/lossFunctionUtils.h"
#include "lossFunctions/categoricalCrossentropy.h"
#include "lossFunctions/sparseCategoricalCrossentropy.h"
#include "lossFunctions/binaryCrossentropy.h"

#include "optimizers/sgd.h"

//#include "H5Cpp.h"
// ---------------------------------------------------------------------------

namespace NilDa
{

neuralNetwork::neuralNetwork():
  layers_(0),
  numberOfLayers_(0),
  lastLayer_(0),
  validState_(false),
  finalizedNetwork_(false)
{}

neuralNetwork::neuralNetwork(const std::vector<layer*>& vLayers):
  layers_(vLayers),
  numberOfLayers_(vLayers.size()),
  lastLayer_(numberOfLayers_ - 1),
  validState_(false),
  finalizedNetwork_(false)
{
  initLayers(/*resetWeights= */ true);
}

void neuralNetwork::initLayers(const bool resetWeightBiases) const
{
  // The first layer must be an input layer
  if (layers_[inputLayer_]->layerType() != layerTypes::input)
  {
     std::cerr << "First layer is not an input layer.\n";

     std::abort();
  }

  layers_[inputLayer_]->checkInput();

  // The last layer must be a dense layer for the output
  if (layers_[lastLayer_]->layerType() != layerTypes::dense)
  {
     std::cerr << "Last layer must be a dense layer.\n";
     
     std::abort();
  }

  // Initialize the hidden and the output layers
  // in the forward direction
  for (int i = firstLayer_; i < numberOfLayers_; ++i)
  {
    layers_[i]->checkInput();

    layers_[i]->init(layers_[i - 1], resetWeightBiases);
  }

  // Setup additional paramters in backward direction
  // for the hidden layers
  for (int i = lastLayer_ - 1; i >= firstLayer_; --i)
  {
    layers_[i]->setupBackward(layers_[i + 1]);
  }
}

void neuralNetwork::summary() const
{
  std::cout << "============================================================\n";
  std::cout << " Layer  Type         Output size     Parameters   Trainable \n";
  std::cout << "============================================================\n";

  int totalParameters = 0;
  int totalTrainableParameters = 0;

  for (int i = 0; i < numberOfLayers_; ++i)
  {
    std::cout << std::left << std::setfill(' ')
              << std::setw(3) << " "
              << i << "    ";

    std::cout << std::setw(12) << std::left
              << layers_[i]->name() << " ";

    const layerSizes sizes = layers_[i]->size();

    if (sizes.isFlat)
    {
      std::cout << std::setw(12) << sizes.size << "    ";
    }
    else
    {
      std::cout << std::setw(16)
                << std::to_string(sizes.rows) + ", " +
                   std::to_string(sizes.cols) + ", " +
                   std::to_string(sizes.channels);
    }

    std::cout << std::setw(14)
              << layers_[i]->numberOfParameters()
              << " ";

    totalParameters += layers_[i]->numberOfParameters();

    if (layers_[i]->isTrainable())
    {
      if (layers_[i]->numberOfParameters() > 0)
      {
        totalTrainableParameters += layers_[i]->numberOfParameters();

        std::cout << "Yes";
      }
      else
      {
        std::cout << "-";
      }
    }
    else
    {
      if (layers_[i]->numberOfParameters() > 0)
      {
        std::cout << "NO";
      }
      else
      {
        std::cout << "-";
      }
    }

    std::cout << "\n";
    if (i != lastLayer_)
    {
      std::cout << "------------------------------------------------------------\n";
    }
  }

  std::cout << "============================================================\n";

  std::cout << "Total parameters: "
            << totalParameters << std::endl;

  std::cout << "Total trainable parameters: "
            << totalTrainableParameters << std::endl;

  std::cout << "Non-trainable parameters: "
            << totalParameters - totalTrainableParameters << "\n\n";

}

void neuralNetwork::configure(
                              const optimizer& opt,
                              const std::string& lossName
                             )
{
  if (numberOfLayers_ <= 0)
  {
    std::cerr << "The neural network has no layer." << std::endl;
    std::abort();
  }

  optimizer_ = &opt;

  initOptimizer();

  setLossFunction(lossFunctionCode(lossName));

  finalizedNetwork_ = true;
}

void neuralNetwork::initOptimizer() const
{
  for (int i = firstLayer_; i < numberOfLayers_; ++i)
  {
    if (layers_[i]->numberOfParameters() > 0 &&
        layers_[i]->isTrainable())
    {
        optimizer_->init(
                         layers_[i]->getWeightsDerivative(),
                         layers_[i]->getBiasesDerivative()
                        );
    }
  }
}

void neuralNetwork::setLossFunction(const std::string& lossName)
{
  setLossFunction(lossFunctionCode(lossName));
}

void neuralNetwork::setLossFunction(const lossFunctions lossCode)
{
  switch(lossCode)
  {
    case lossFunctions::categoricalCrossentropy :
        // add check size output layer > 2
        lossFunction_ = std::make_unique<categoricalCrossentropy>();
        break;
    case lossFunctions::sparseCategoricalCrossentropy :
        lossFunction_ = std::make_unique<sparseCategoricalCrossentropy>();
        break;
    case lossFunctions::binaryCrossentropy :
        lossFunction_ = std::make_unique<binaryCrossentropy>();
        break;
    default :
        std::cerr << "Not valid loss function  "
                    << lossFunctionName(lossCode)
                    << " in this context.\n";
    std::abort();
  }
}

void neuralNetwork::forwardPropagation(
                                       const Matrix& obs,
                                       const bool trainingPhase
                                      ) const
{
  validState_ = false;

#ifdef ND_DEBUG_CHECKS
  // The layer 0 is an input layer, just check that
  // the size of the input data is consistent with the
  // input layer size

  layers_[inputLayer_]->checkInputSize(obs);
#endif

  // The first actual layer takes in directly the input data
  layers_[firstLayer_]->forwardPropagation(obs, trainingPhase);

  for (int i = firstLayer_ + 1; i < numberOfLayers_; ++i)
  {
    // The other layers take in the output of the previous layer
    layers_[i]->forwardPropagation(
                                   layers_[i - 1]->output(),
                                   trainingPhase
                                  );
  }

  validState_ = true;
}

void neuralNetwork::backwardPropagation(
                                        const Matrix& obs,
                                        const Matrix& labels
                                       ) const
{
#ifdef ND_DEBUG_CHECKS
  assert(validState_);

  assert(lossFunction_);
#endif

  // Derivative of the cost function w.r.t the activation
  // output of the last layer
  Matrix dLoss;
  //dLoss.resize(labels.rows(), labels.cols();

  lossFunction_->computeDerivative(
                                   layers_[lastLayer_]->output(),
                                   labels,
                                   dLoss
                                  );

  // Derivatives of the cost function w.r.t. the
  // weights and biases of the last layer
  layers_[lastLayer_]->backwardPropagation(
                                           dLoss,
                                           layers_[lastLayer_ - 1]->output()
                                          );

  // Maybe the if statement is not necessary
  if(numberOfLayers_ > 2)
  {
    // Derivatives of the cost function w.r.t. the
    // weights and biases for the hidden layers
    for (int i = lastLayer_ - 1; i > firstLayer_; --i)
    {
      layers_[i]->backwardPropagation(
                                      layers_[i + 1]->backPropCache(),
                                      layers_[i - 1]->output()
                                     );
    }
  }

  // Derivatives of the cost function w.r.t. the
  // weights and biases of the first layer
  layers_[firstLayer_]->backwardPropagation(
                                            layers_[firstLayer_ + 1]->backPropCache(),
                                            obs
                                           );
}

Scalar neuralNetwork::getLoss(const Matrix& labels) const
{
#ifdef ND_DEBUG_CHECKS
  assert(validState_);
#endif

  return lossFunction_->compute(
                                layers_[lastLayer_]->output(),
                                labels
                               );
}

Scalar neuralNetwork::propagate(
                                const Matrix& obs,
                                const Matrix& labels,
                                const bool trainingPhase
                               ) const
{
  forwardPropagation(obs, trainingPhase);

  backwardPropagation(obs, labels);

  return getLoss(labels);
}

void neuralNetwork::train(
                          const Matrix& obs,
                          const Matrix& labels,
                          const int epochs,
                          const int batchSize,
                          const int verbosity
                         ) const
{
  if (!finalizedNetwork_)
  {
    std::cerr << "The neural network has not been configured." << std::endl;

    std::abort();
  }

  assert(epochs > 0);

  assert(batchSize > 0);

  const int nObs = obs.cols()
                 / layers_[inputLayer_]->inputStride();

  assert(batchSize <= nObs);

  const layerSizes sizes = layers_[lastLayer_]->size();

  lossFunction_->sanityCheck(sizes.size, labels);

  const int epochSteps = floor((float)nObs/(float)batchSize);

  const int batchStride = batchSize
                        * layers_[inputLayer_]->inputStride();

#ifdef ND_DEBUG_CHECKS
  assert(epochSteps > 0);
  assert(batchStride > 0);
#endif

  for (int i = 0; i < epochs; ++i)
  {
    std::cout <<  "Epoch " << i+1 << "/" << epochs << std::endl;

    progressBar progBar;

    auto startTime = std::chrono::high_resolution_clock::now();

    Scalar totErr = 0;

    int totN = 0;

    for (int j = 0; j < epochSteps; ++j)
    {
      ConstMapMatrix obsBatched(
                                obs(
                                    Eigen::all,
                                    Eigen::seqN(j*batchStride, batchStride)
                                  ).data(),
                                obs.rows(),
                                batchStride
                               );

     ConstMapMatrix labelsBatched(
                                  labels(
                                         Eigen::all,
                                         Eigen::seqN(j*batchStride, batchStride)
                                       ).data(),
                                  labels.rows(),
                                  batchStride
                                 );

      Scalar loss = propagate(
                              obsBatched,
                              labelsBatched,
                              /*trainingPhase=*/true
                             );

      update();

      totErr += getSumError(
                            obsBatched,
                            labelsBatched,
                            /*runForward=*/false
                           );

      totN += labelsBatched.size();

      if (verbosity > 1)
      {
        const float progess = (float)(j+1) / epochSteps;

				const std::string message = std::to_string(j+1)
                                  + "/"
                                  + std::to_string(epochSteps)
      			       		            + " Cost function: "
                                  + std::to_string(loss);

        progBar.update(progess, message);
      }
    }

    progBar.close();

    auto stopTime = std::chrono::high_resolution_clock::now();

    auto elapsedTime =
      std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime);

    Scalar accuracy = 1.0 - (totErr/totN);

    std::cout << "Accuracy: " << accuracy
              << " -- "
   	          << 0.001*elapsedTime.count() << " s: "
   	          << (float)elapsedTime.count()/epochSteps << " ms/steps\n";
  }
}

void neuralNetwork::update() const
{
  Matrix deltaWeights;

  Vector deltaBiases;

  for (int i = firstLayer_; i < numberOfLayers_; ++i)
  {
    if (layers_[i]->numberOfParameters() > 0 &&
        layers_[i]->isTrainable())
    {
      optimizer_->update(
                         layers_[i]->getWeightsDerivative(),
                         layers_[i]->getBiasesDerivative(),
                         deltaWeights,
                         deltaBiases
                        );

      layers_[i]->incrementWeightsAndBiases(
                                            deltaWeights,
                                            deltaBiases
                                           );
    }
  }
}

void neuralNetwork::predict(
                            const Matrix& obs,
                            Matrix& predictions,
                            const bool runForward
                           ) const
{
  if(runForward)
  {
    forwardPropagation(obs, /*trainingPhase=*/false);
  }

  lossFunction_->predict(
                         layers_[lastLayer_]->output(),
                         predictions
                        );
}

Scalar neuralNetwork::getSumError(
                                  const Matrix& obs,
                                  const Matrix& trueData,
                                  const bool runForward
                                 ) const
{
  Matrix predictions;

  predict(obs, predictions, runForward);

  Matrix err = predictions - trueData;

  return (err.array().abs().colwise().sum() > 0).count();
}

Scalar neuralNetwork::getAccuracy(
                                  const Matrix& obs,
                                  const Matrix& trueData
                                 ) const
{
  // Sum of all the errors
  Scalar err = getSumError(obs, trueData, /*runForward=*/ true);

  // Mean error
  return 1.0 - (err / trueData.cols());
}

Scalar neuralNetwork::getAccuracy(
                                  const Matrix& obs,
                                  const Matrix& trueData,
                                  const int batchSize
                                 ) const
{
  const int batchStride = batchSize
                        * layers_[inputLayer_]->inputStride();

  const int nObs = obs.cols()
                 / layers_[inputLayer_]->inputStride();

  const int nObsBatch = floor((float)nObs/(float)batchSize);

  Scalar totErr = 0;

  int totN = 0;

  for (int j = 0; j < nObsBatch; ++j)
  {
    totErr += getSumError(
                          obs(
                              Eigen::all,
                              Eigen::seqN(j*batchStride, batchStride)
                             ),
                          trueData(
                                   Eigen::all,
                                   Eigen::seqN(j*batchStride, batchStride)
                                  ),
                          /*runForward=*/ true
                         );

    totN += batchStride;
  }

  return 1.0 - (totErr / totN);
}

void neuralNetwork::getProbability(
                                   const Matrix& obs,
                                   Matrix& probability
                                  )
{
  forwardPropagation(obs, /*trainingPhase=*/false);

  probability = layers_[lastLayer_]->output();
}

void neuralNetwork::saveModel(std::string outputFile) const
{
  std::ofstream ofs(
                    outputFile,
                    std::ofstream::out |
                    std::ofstream::binary |
                    std::ofstream::trunc
                   );

  if (ofs.fail())
  {
    std::cerr << "Impossible to save the model to "
              << outputFile << "\n";

    std::abort();
  }

  ofs.write((char*) (&numberOfLayers_ ), sizeof(int));

  for (int i = 0; i < numberOfLayers_; ++i)
  {
    layers_[i]->saveLayer(ofs);
  }

  const int lossCode = lossFunction_->type();
  ofs.write((char*) (&lossCode), sizeof(int));

  ofs.close();
}

void neuralNetwork::loadModel(std::string inputFile)
{
  std::ifstream ifs(
                    inputFile,
                    std::ofstream::in |
                    std::ofstream::binary
                   );

  if (ifs.fail())
  {
    std::cerr << "Impossible to load model from "
              << inputFile << "\n";

    std::abort();
  }

  ifs.read((char*) (&numberOfLayers_), sizeof(int));

  lastLayer_ = (numberOfLayers_ - 1);

  for (int i = 0; i < numberOfLayers_; ++i)
  {
    int layerType;
    ifs.read((char*) (&layerType), sizeof(int));

    layers_.push_back(createLayer(layerType));

    layers_[i]->loadLayer(ifs);

    layers_[i]->checkInput();
  }

  initLayers(/*resetWeights= */ false);

  int lCode;
  ifs.read((char*) (&lCode), sizeof(int));

  lossFunctions lType = static_cast<lossFunctions>(lCode);

  setLossFunction(lType);

  ifs.close();
}

errorCheck
neuralNetwork::checkWeightsGradients(
                                     const int layer,
                                     const Matrix& obs,
                                     const Matrix& labels,
                                     const Scalar eps,
                                     const Scalar errorLimit
                                    ) const
{
  // Store the original weights and biases
  const Matrix weightsBk = layers_[layer]->getWeights();

  const Vector biasesBk = layers_[layer]->getBiases();

  Matrix weights(weightsBk.rows(), weightsBk.cols());

  Matrix dWeightsNum(weights.rows(), weights.cols());

  for (int i = 0; i < weights.rows(); ++i)
  {
    for (int j = 0; j < weights.cols(); ++j)
    {
      // Compute the W+eps part
      weights.noalias() = weightsBk;
      weights(i, j) += eps;

      layers_[layer]->setWeightsAndBiases(
                                          weights,
                                          biasesBk
                                         );

      forwardPropagation(obs);

      const Scalar Jp = getLoss(labels);

      // Compute the W-eps part
      weights.noalias() = weightsBk;
      weights(i, j) -= eps;

      layers_[layer]->setWeightsAndBiases(
                                          weights,
                                          biasesBk
                                         );

      forwardPropagation(obs);

      const Scalar Jm = getLoss(labels);

      // Numerical gradients: central difference
      dWeightsNum(i,j) = (Jp - Jm)/(2*eps);
    }
  }

  // Restore the correct values of weights and biases
  layers_[layer]->setWeightsAndBiases(
                                      weightsBk,
                                      biasesBk
                                     );

  // Analytical derivative of the weights
  const Matrix dWeights = layers_[layer]->getWeightsDerivative();

  const Scalar error = (dWeights - dWeightsNum).norm()
                     / (dWeights.norm() + dWeightsNum.norm());

  errorCheck output;
  output.code = (error < errorLimit) ? EXIT_OK : EXIT_FAIL;
  output.error = error;

  return output;
}

errorCheck
neuralNetwork::checkBiasesGradients(
                                    const int layer,
                                    const Matrix& obs,
                                    const Matrix& labels,
                                    const Scalar eps,
                                    const Scalar errorLimit
                                   ) const
{
  // Store the original weights and biases
  const Matrix weightsBk = layers_[layer]->getWeights();

  const Vector biasesBk = layers_[layer]->getBiases();

  Vector biases(biasesBk.rows());

  Vector dBiasesNum(biases.rows());

  for (int i = 0; i < biases.rows(); ++i)
  {
    // Compute the W+eps part
    biases.noalias() = biasesBk;
    biases(i) += eps;

    layers_[layer]->setWeightsAndBiases(
                                        weightsBk,
                                        biases
                                       );

    forwardPropagation(obs);

    const Scalar Jp = getLoss(labels);

    // Compute the W-eps part
    biases.noalias() = biasesBk;
    biases(i) -= eps;

    layers_[layer]->setWeightsAndBiases(
                                        weightsBk,
                                        biases
                                       );

    forwardPropagation(obs);

    const Scalar Jm = getLoss(labels);

    // Numerical gradients: central difference
    dBiasesNum(i) = (Jp - Jm)/(2*eps);
  }

  // Restore the correct values of weights and biases
  layers_[layer]->setWeightsAndBiases(
                                      weightsBk,
                                      biasesBk
                                     );

  // Analytical derivative of the biases
  const Vector dBiases = layers_[layer]->getBiasesDerivative();

  const Scalar error = (dBiases - dBiasesNum).norm()
                     /  (dBiases.norm() + dBiasesNum.norm());

  errorCheck output;
  output.code = (error < errorLimit) ? EXIT_OK : EXIT_FAIL;
  output.error = error;

  return output;
}

int neuralNetwork::gradientsSanityCheck(
                                        const Matrix& obs,
                                        const Matrix& labels,
                                        const bool printError
                                       ) const
{
#ifdef ND_SP
  #warning "Single precision used. For testing specify either double or long precision."
#endif

  int code = EXIT_OK;

  forwardPropagation(obs);

  backwardPropagation(obs, labels);

  const Scalar errorLimit = 1.0e-8;

  const Scalar eps = 1.0e-5;

  for(int i = lastLayer_; i >= firstLayer_; --i)
  {
    if (layers_[i]->numberOfParameters() > 0)
    {
      const errorCheck outputW =
        checkWeightsGradients(i, obs, labels, eps, errorLimit);

      const errorCheck outputB =
        checkBiasesGradients(i, obs, labels, eps, errorLimit);

      if(outputW.code == EXIT_FAIL || outputW.code == EXIT_FAIL)
      {
        code = EXIT_FAIL;
      }

      if(printError || code == EXIT_FAIL)
      {
        std::cout << "Layer: "
                  << layers_[i]->name() << "\n"
                  << "Error weights = "
                  << outputW.error << " "
                  << ", Error biases = "
                  << outputB.error << "\n\n";
      }
    }
  }

  return code;
}

} // namespace
