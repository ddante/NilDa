#include <iostream>
#include <assert.h>

#include "activationFunctionUtils.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


activationFucntions
activationFunctionCode(const std::string& inName)
{
  if (inName == "identity")
  {
    return activationFucntions::identity;
  }
  else if (inName == "sigmoid")
  {
    return activationFucntions::sigmoid;
  }
  else if (inName == "relu")
  {
    return activationFucntions::relu;
  }
  else if (inName == "softmax")
  {
    return activationFucntions::softmax;
  }
  else if (inName == "tanh")
  {
    return activationFucntions::tanh;
  }
  else
  {
    std::cerr << "Unknown activation function name "
              << inName << "\n";
    assert(false);
  }
}

std::string
activationFunctionName(const activationFucntions type)
{
  if (type == activationFucntions::identity)
  {
    return "identity";
  }
  else if (type == activationFucntions::sigmoid)
  {
    return "sigmoid";
  }
  else if (type == activationFucntions::relu)
  {
    return "relu";
  }
  else if (type == activationFucntions::softmax)
  {
    return "softmax";
  }
  else if (type == activationFucntions::tanh)
  {
    return "tanh";
  }
  else
  {
    std::cerr << "Invalid activation function code.\n";
    assert(false);
  }
}


} // namespace
