#include <iostream>
#include <fstream>

#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <tuple>
#include <random>
#include <algorithm>
#include <iterator>

#include "primitives/Matrix.h"
#include "primitives/errors.h"
#include "utils/progressBar.h"

#include "importDatasets.h"

// ---------------------------------------------------------------------------
namespace NilDa
{


int read_dataset(
                 const std::string& path,
                 const bool shuffle,
                 std::vector< std::pair<std::string, int> >& DataSet
                )
{
  DIR *dir;

  class stat buf;

  dir = opendir(path.c_str());

  if (!dir)
  {
    std::cerr << "Impossible to open directory " << dir << ".\n";

    std::abort();
  }

  class dirent *entry;

  while ((entry = readdir(dir)) != NULL)
  {
    const std::string entryName = entry->d_name;

    // Exclude current & previous directories
    if (entryName[0] == '.')
    {
      continue;
    }

    struct stat pathStat;

    stat(entryName, &pathStat);

    // Get the directories
    if (S_ISDIR(pathStat.st_mode))
    {
      std::cout << entryName << "\n";
    }
}


} // namespace
