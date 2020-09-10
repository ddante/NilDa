#include <iostream>

#include <string>

#include "primitives/Matrix.h"

#include "images.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

// ---------------------------------------------------------------------------
namespace NilDa
{

void displayImage(
                  const Matrix& inputImages,
                  const std::array<int, 3>& size,
                  const int id,
                  const std::string& title
                 )
{
  const int rows = size[0];
  const int cols = size[1];
  const int channels = size[2];

  const int imgStride = channels;

  assert(inputImages.rows() == rows * cols);

  assert(inputImages.cols() > (id + 1) * imgStride);

  ConstMapMatrix imgMap(
                        inputImages(
                                    Eigen::all,
                                    Eigen::seqN(id*imgStride, imgStride)
                                   ).data(),
                         rows,
                         cols
                        );

  Matrix img(imgMap);

  cv::Mat cvImg;
  cv::eigen2cv(img, cvImg);

  cv::resize(cvImg, cvImg, cv::Size(100, 100));

  cv::imshow(title, cvImg);

  cv::waitKey(1000);
}

} // namepsace
