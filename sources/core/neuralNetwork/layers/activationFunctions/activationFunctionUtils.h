#ifndef ACTVIVATION_FUNCTION_UTILS_H
#define ACTVIVATION_FUNCTION_UTILS_H

#include <string>

// ---------------------------------------------------------------------------

namespace NilDa
{


enum class activationFucntions
{
  identity,
  sigmoid,
  relu,
  softmax,
  tanh,
};

// Return the enum name of the activation function from the string name
activationFucntions
activationFunctionCode(const std::string& inName);


} // namespace

#endif
