#ifndef ERRORS_H
#define ERRORS_H

#include "Scalar.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


const int EXIT_OK = 0;

const int EXIT_FAIL = 1;

struct errorCheck
{
  int code;
  Scalar error;
};


} // namespace

#endif
