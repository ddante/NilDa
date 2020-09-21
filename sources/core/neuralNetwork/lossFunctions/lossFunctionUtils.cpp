#include <iostream>
#include <assert.h>

#include "lossFunctionUtils.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

lossFunctions lossFunctionCode(const std::string& inName)
{
  if (inName == "categorical_crossentropy")
  {
    return lossFunctions::categoricalCrossentropy;
  }
  else if (inName == "sparse_categorical_crossentropy")
  {
    return lossFunctions::sparseCategoricalCrossentropy;
  }
  else if (inName == "binary_crossentropy")
  {
    return lossFunctions::binaryCrossentropy;
  }
  else
  {
    std::cerr << "Unknown loss function name "
              << inName << ".\n";

    std::abort();
  }
}

std::string lossFunctionName(const lossFunctions type)
{
  if (type == lossFunctions::categoricalCrossentropy)
  {
    return "categorical_crossentropy";
  }
  else if (type == lossFunctions::sparseCategoricalCrossentropy)
  {
    return "sparse_categorical_crossentropy";
  }
  else if (type == lossFunctions::binaryCrossentropy)
  {
    return "binary_crossentropy";
  }
  else
  {
    std::cerr << "Unknown loss function code.\n";

    std::abort();
  }
}

} // namespace
