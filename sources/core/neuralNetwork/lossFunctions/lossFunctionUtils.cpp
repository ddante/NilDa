#include <iostream>
#include <assert.h>

#include "lossFunctionUtils.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

lossFunctions lossFunctionCode(const std::string& inName)
{
  if(inName == "sparse_categorical_crossentropy")
  {
    return lossFunctions::sparseCategoricalCrossentropy;
  }
  else
  {
    std::cerr << "Unknown loss function name "
              << inName << ".\n";

    assert(false);
  }
}

std::string lossFunctionName(const lossFunctions type)
{
  if(type == lossFunctions::sparseCategoricalCrossentropy)
  {
    return "sparse_categorical_crossentropy";
  }
  else
  {
    std::cerr << "Unknown loss function code.\n";

    assert(false);
  }
}

} // namespace
