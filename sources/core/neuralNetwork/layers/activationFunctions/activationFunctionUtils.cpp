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
              << inName
              << "." << std::endl;
    assert(false);
  }
}


} // namespace
