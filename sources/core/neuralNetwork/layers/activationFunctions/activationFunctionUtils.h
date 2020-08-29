#ifndef ACTVIVATION_FUNCTION_UTILS_H
#define ACTVIVATION_FUNCTION_UTILS_H

#include <string>

// ---------------------------------------------------------------------------

namespace NilDa
{


enum class activationFunctions
{
  identity,
  sigmoid,
  relu,
  softmax,
  tanh,
};

// Return the enum name of the activation function from the string name
activationFunctions
activationFunctionCode(const std::string& inName);

// Return the string name of the activation function from enum name
std::string
activationFunctionName(const activationFunctions type);

} // namespace

#endif
