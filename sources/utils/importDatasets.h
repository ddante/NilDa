#ifndef IMPORT_DATASETS_H
#define IMPORT_DATASETS_H

#include <iostream>
#include <fstream>

#include <string>

#include "primitives/Matrix.h"

// ---------------------------------------------------------------------------

namespace NilDa
{


int importMNISTDatabase(
                        const std::string& fullPathFileImage,
	                      const std::string& fullPathFileLabel,
	                      const double imageScaling,
	                      const bool shuffle,
                        Matrix& Images,
                        Matrix& Labels
                      );


} // namespace

#endif
