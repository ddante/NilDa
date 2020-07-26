#include <iostream>
#include <fstream>

#include <string>

#include "primitives/Matrix.h"
#include "primitives/errors.h"
#include "utils/progressBar.h"

#include "importDatasets.h"

// ---------------------------------------------------------------------------
namespace NilDa
{


static int reverseInt(int i)
{
  unsigned char c1, c2, c3, c4;

  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;

  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


int importMNISTDatabase(
                        const std::string& fullPathFileImage,
                        const std::string& fullPathFileLabel,
                        const double imageScaling,
                        const bool shuffle,
                        Matrix& Images,
                        Matrix& Labels
)
{
  std::cout << "Importing MNIST training data" << std::endl;

  // Open the file of of the images
  std::ifstream fileImages;

  fileImages.open(fullPathFileImage);

  if (!fileImages.is_open())
  {
    std::cerr << "Impossible to open file " <<  fullPathFileImage << std::endl;
    return EXIT_FAIL;
  }

  // Open the file of the labes
  std::ifstream fileLabels;

  fileLabels.open(fullPathFileLabel);

  if (!fileLabels.is_open())
  {
    std::cerr << "Impossible to open file " <<  fullPathFileLabel << std::endl;
    return EXIT_FAIL;
  }

  // Read the magic number
  int magicNumber = 0;
  fileImages.read( (char*)&magicNumber, sizeof(magicNumber) );
  magicNumber = reverseInt(magicNumber);

  fileLabels.read( (char*)&magicNumber, sizeof(magicNumber) );
  magicNumber = reverseInt(magicNumber);

  // Read the number of images
  int numberOfImages = 0;
  fileImages.read( (char*)&numberOfImages, sizeof(numberOfImages) );
  numberOfImages = reverseInt(numberOfImages);

  // Read the number of labels
  int numberOfLabels = 0;
  fileLabels.read( (char*)&numberOfLabels, sizeof(numberOfLabels) );
  numberOfLabels = reverseInt(numberOfLabels);

  if (numberOfLabels != numberOfImages)
  {
    std::cerr << "Number of images ("
              << numberOfImages  << ") "
              << "does not match number of labels ("
              << numberOfLabels  << ")."
              << std::endl;

     return EXIT_FAIL;
  }

  std::cout << "Number of images: " << numberOfImages << std::endl;

  // Read the number of rows and colums of each image
  int numberOfRows = 0;
  fileImages.read((char*)&numberOfRows, sizeof(numberOfRows));
  numberOfRows = reverseInt(numberOfRows);

  int numberOfCols = 0;
  fileImages.read((char*)&numberOfCols, sizeof(numberOfCols));
  numberOfCols = reverseInt(numberOfCols);

  std::cout << "Size of the image: "
            << numberOfRows
            << " x "
            << numberOfCols
            << std::endl;

  const int imgSize = numberOfRows*numberOfCols;

  Images.resize(imgSize, numberOfImages);

  // Classes [0, ..., 9]
  const int numberOfClasses = 10;

  Labels.resize(numberOfClasses, numberOfImages);
  Labels.setZero(numberOfClasses, numberOfImages);

  progressBar progBar;

  // Read each label and the corresponding image
  for (int i = 0; i < numberOfImages; ++i)
  {
    char label;
    fileLabels.read((char*)&label, sizeof(label));

    int idx = 0;

    for (int r = 0; r < numberOfRows; ++r)
    {
      for (int c = 0; c < numberOfRows; ++c)
      {
        unsigned char pixel;
        fileImages.read((char*)&pixel, sizeof(pixel));

        Images(idx, i) = static_cast<Scalar>(pixel);

        idx++;
      }
    }

    // One hot encoding of the number classes
    Labels(static_cast<int>(label), i) = 1;

    progBar.update(i+1, numberOfImages);
  }

  progBar.close();

  std::cout << "Scaling images with parameter: "
            << imageScaling << std::endl;

  // Scale the images with the input scaling parameter
  Images *= imageScaling;

  // Shouffle the images with the corresponding labels
  if (shuffle)
  {
    std::cout << "Shuffling the datasets" << std::endl;

    Eigen::PermutationMatrix<
                             Eigen::Dynamic,
                             Eigen::Dynamic
                            >
                            perm(Images.cols());

     perm.setIdentity();

     std::random_shuffle(
                         perm.indices().data(),
                         perm.indices().data() + perm.indices().size()
                        );

    // Permute columns
    Images = Images * perm;
    Labels = Labels * perm;
  }

  fileImages.close();
  fileLabels.close();

  return EXIT_OK;
}


} // namespace
