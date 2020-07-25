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
  const optimizer* optimizer_;

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

  // Initialize the optimizer internal cache for each layer
  void initOptimizer() const;

  // Update the weights and biases of each layer
  // with the increments computed by the optimizer
  void update() const;

public:

    // Constructor

    neuralNetwork() = delete;

    neuralNetwork(const std::vector<layer*>& vectorLayer);

    // Member functions

    //void addLayer(const layer singleLayer);

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
               const int batchSize
              ) const;

    // Set the loss function
    void setLossFunction(const std::string& lossName);

    // Return the value of the loss function at the current state
    Scalar getLoss(const Matrix& obs, const Matrix& labels) const;

    // Check if the analytical gradients matches numerical ones
    int gradientsSanityCheck(
                             const Matrix& obs,
                             const Matrix& label,
                             const bool printError
                            ) const;

    // Destructor

    ~neuralNetwork()
    {
      for (int i = 0; i < numberOfLayers_; ++i)
      {
        delete layers_[i];
      }
    };

};


} // namespace

#endif
