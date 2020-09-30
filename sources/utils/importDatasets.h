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
                        const bool sparseCategorical,
                        Matrix& Images,
                        Matrix& Labels
                      );

int readImagedataset(
                     const std::string& path,
                     const bool shuffle,
                     std::vector< std::pair<std::string, int> >& DataSet
                    );

} // namespace

#endif
