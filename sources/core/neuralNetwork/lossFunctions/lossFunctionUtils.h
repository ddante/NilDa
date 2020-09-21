#ifndef LOSS_FUNCTION_UTILS_H
#define LOSS_FUNCTION_UTILS_H

namespace NilDa
{


enum class lossFunctions
{
  categoricalCrossentropy,
  sparseCategoricalCrossentropy,
  binaryCrossentropy
};

lossFunctions lossFunctionCode(const std::string& inName);

std::string lossFunctionName(const lossFunctions type);


} // namespace

#endif
