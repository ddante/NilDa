#ifndef LAYER_H
#define LAYER_H

#include <assert.h>
#include <array>

#include "primitives/Matrix.h"
#include "primitives/Vector.h"
#include "primitives/errors.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

enum class layerTypes {
  input,
  dense,
  conv2D,
  maxPool2D
};

struct layerSizes {
  bool isFlat;
  int size;
  int rows;
  int cols;
  int channels;
};

class layer
{

protected:

  layerSizes size_;

  layerTypes type_;

  bool trainable_;

public:

  // Constructors

  layer(){}

  // Member functions

  // Initialize the layer
  virtual void init(const layer* previousLayer) = 0;

  // Setup additional paramters in backward direction
  // for the hidden layers
  virtual void setupBackward(const layer* nextLayer) = 0;

  // Check if the dimensions of input data are consistent with the layer
  virtual void checkInputSize(const Matrix& inputData) const = 0;

  // Forward step for the layer.
  // It takes as input the output of the previous layer
  // and computes the activation of the current layer
  virtual void forwardPropagation(const Matrix& inputData) = 0;

  // Backward step for the layer.
  // It takes as inputs the back propagation cache of
  // the next layer and the output of the previous layer.
  // It computes the derivative of the loss w.r.t. the
  // weights and biases of the current layer and stores
  // the cache to be used by the previous layer
  virtual void backwardPropagation(
                                   const Matrix& dActivationNext,
                                   const Matrix& inputData
                                  ) = 0;

  // Return the weight matrix of the layer
  virtual const Matrix& getWeights() const = 0;

  // Return the biase vector of the layer
  virtual const Vector& getBiases() const = 0;

  // Return the weight derivative matrix of the layer
  virtual const Matrix& getWeightsDerivative() const = 0;

  // Return the bias derivative vector of the layer
  virtual const Vector& getBiasesDerivative() const = 0;

  // Return the activation of the layer
  virtual const Matrix& output() const = 0;

  // Return the matrix with the bacpropagation cache
  // for the backward propagation
  virtual const Matrix& backPropCache() const = 0;

  // Set the values of the weights and biases in the layer
  virtual void setWeightsAndBiases(
                                   const Matrix& W,
                                   const Vector& b
                                  ) = 0;

   // Update the weights and biases in the layer with an increment
   virtual void incrementWeightsAndBiases(
                                          const Matrix& deltaW,
                                          const Vector& deltaB
                                         ) = 0;

  // Perform local checks in the layers for debugging
  virtual errorCheck localChecks(
                                 const Matrix& input,
                                 Scalar errTol
                                ) const = 0;

  // Return the colum stride for each observation in the input
  virtual int inputStride() const = 0;

  //virtual void update() = 0;

  layerTypes layerType() const
  {
    return type_;
  }

  layerSizes size() const
  {
    return size_;
  }

  bool isTrainable() const
  {
    return trainable_;
  }

  // Return the string name of the type layer from the enum name
  std::string getLayerName(const layerTypes inLayerType) const
  {
    std::string name;

    if (inLayerType == layerTypes::input)
    {
      name = "Input";
    }
    else if (inLayerType == layerTypes::dense)
    {
      name = "Dense";
    }
    else if (inLayerType == layerTypes::conv2D)
    {
      name = "Conv 2D";
    }
    else if (inLayerType == layerTypes::maxPool2D)
    {
      name = "MaxPool 2D";
    }
    else
    {
      std::cerr << "Unknown layer type code." << std::endl;
      assert(false);
    }

    return name;
  }

  // Return the string name of the type layer from the enum name
  std::string layerName() const
  {
    return getLayerName(type_);
  }

  virtual int numberOfParameters() const = 0;

  // Destructor
  virtual ~layer()  = default;
};


} // namespace

#endif
