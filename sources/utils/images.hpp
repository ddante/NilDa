#ifndef IMAGES_H
#define IMAGES_H

#include <string>

#include "primitives/Matrix.h"

// ---------------------------------------------------------------------------

namespace NilDa
{

void displayImage(
                  const Matrix& inputImages,
                  const std::array<int, 3>& size,
                  const int id
                 );


} // namespace

#endif
