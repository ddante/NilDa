#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <memory>

#include "primitives/errors.h"

#include "primitives/Scalar.h"
#include "primitives/Vector.h"
#include "primitives/Matrix.h"

#include "layers/layer.h"

#include "optimizers/optimizer.h"

#include "lossFunctions/lossFunction.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


class neuralNetwork
{

private:

  // List of all the layer in the neural network
  std::vector<layer*> layers_;

  // Pointer to the loss function
  std::unique_ptr<lossFunction> lossFunction_;

  // Pointer to the optimizer
  const optimizer* optimizer_ = nullptr;

  // Total number of layers (input + hidden + output)
  int numberOfLayers_;

  // Store for simplicity the number of the last layer:
  // numberOfLayers_ - 1
  int lastLayer_;

  mutable bool validState_;

  mutable bool finalizedNetwork_;

  // Number of the input layer
  const int inputLayer_ = 0;

  // Number of the first layer after of the input layer
  const int firstLayer_ = 1;

private:

  // Internal routines to compute the difference between
  // the analytical and the numerical gradients
  errorCheck
  checkWeightsGradients(
                        const int layer,
                        const Matrix& obs,
                        const Matrix& labels,
                        const Scalar eps,
                        const Scalar errorLimit
                       ) const;

  errorCheck
  checkBiasesGradients(
                       const int layer,
                       const Matrix& obs,
                       const Matrix& labels,
                       const Scalar eps,
                       const Scalar errorLimit
                      ) const;

  // Initialize the layers of the network
  void initLayers(const bool resetWeightBiases) const;

  // Initialize the optimizer internal cache for each layer
  void initOptimizer() const;

  // Perform one forward and backaward step
  Scalar propagate(const Matrix& obs, const Matrix& labels) const;

  // Update the weights and biases of each layer
  // with the increments computed by the optimizer
  void update() const;

public:

  // Constructor

  neuralNetwork();

  neuralNetwork(const std::vector<layer*>& vectorLayer);

  // Member functions

  //void addLayer(const layer singleLayer);

  void summary() const;

  void forwardPropagation(const Matrix& obs) const;

  void backwardPropagation(
                           const Matrix& obs,
                           const Matrix& label
                          ) const;

  void configure(
                 const optimizer& opt,
                 const std::string& lossName
                );

  void train(
             const Matrix& obs,
             const Matrix& labes,
             const int epochs,
             const int batchSize,
             const int verbosity = 2
            ) const;

  // Set the loss function
  void setLossFunction(const lossFunctions lossCode);
  void setLossFunction(const std::string& lossName);

  // Return the value of the loss function at the current state
  Scalar getLoss(const Matrix& labels) const;

  // Compute the prediciton for the given data
  void predict(
               const Matrix& obs,
               Matrix& predictions,
               const bool runForward = true
              ) const;

  // Compute the difference between the prediction and
  // the true data. The return value is the sum of all the errors
  Scalar getSumError(
                     const Matrix& predictions,
                     const Matrix& trueData,
                     const bool runForward = true
                    ) const;

  // Return the accuracy of the trained model
  // using the whole dataset at once
  Scalar getAccuracy(const Matrix& obs, const Matrix& trueData) const;

  // Return the accuracy of the trained model
  // splitting the dataset in batches
  Scalar getAccuracy(
                     const Matrix& obs,
                     const Matrix& trueData,
                     const int batchSize
                    ) const;

  // Save the trained model and the weights to a file
  void saveModel(std::string outputFile) const;

  // Load the model and the weights from a file
  void loadModel(std::string inputFile);

  // Check if the analytical gradients matches numerical ones
  int gradientsSanityCheck(
                           const Matrix& obs,
                           const Matrix& label,
                           const bool printError
                          ) const;

  // Destructor

  ~neuralNetwork()
  {
    if (layers_.size() > 0)
    {
      for (int i = 0; i < numberOfLayers_; ++i)
      {
        delete layers_[i];
      }
    }
  };

};


} // namespace

#endif
