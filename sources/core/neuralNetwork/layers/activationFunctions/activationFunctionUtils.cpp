#include <iostream>
#include <assert.h>

#include "activationFunctionUtils.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


activationFunctions
activationFunctionCode(const std::string& inName)
{
  if (inName == "identity")
  {
    return activationFunctions::identity;
  }
  else if (inName == "sigmoid")
  {
    return activationFunctions::sigmoid;
  }
  else if (inName == "relu")
  {
    return activationFunctions::relu;
  }
  else if (inName == "softmax")
  {
    return activationFunctions::softmax;
  }
  else if (inName == "tanh")
  {
    return activationFunctions::tanh;
  }
  else
  {
    std::cerr << "Unknown activation function name "
              << inName << "\n";
    assert(false);
  }
}

std::string
activationFunctionName(const activationFunctions type)
{
  if (type == activationFunctions::identity)
  {
    return "identity";
  }
  else if (type == activationFunctions::sigmoid)
  {
    return "sigmoid";
  }
  else if (type == activationFunctions::relu)
  {
    return "relu";
  }
  else if (type == activationFunctions::softmax)
  {
    return "softmax";
  }
  else if (type == activationFunctions::tanh)
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
