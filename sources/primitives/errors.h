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

  void setOK()
  {
    code = EXIT_OK;
    error = 0;
  }

  void setFail()
  {
    code = EXIT_FAIL;
    error = 1;
  }
};


} // namespace

#endif
