#ifndef PROGRESS_BAR_H
#define PROGRESS_BAR_H

#include <iostream>

// ---------------------------------------------------------------------------

namespace NilDa
{

class progressBar
{

private:

  int barWidth_;

public:

  // Constructor

  // Default bar width of 40
  progressBar():
    barWidth_(40)
  {}

  // User defined bar width
  progressBar(const int width):
    barWidth_(width)
  {}

  // Member functions

  template<typename N, typename S>
  void update(const N& progress, const S& message) const
  {
    std::cout << "[";

    const int pos = barWidth_ * progress;

    for (int k = 0; k < barWidth_; ++k)
    {
      if (k < pos)
      {
        std::cout << "=";
      }
      else if (k == pos)
      {
        std::cout << ">";
      }
      else
      {
        std::cout << " ";
      }
    }

    std::cout << "] "
              << int(progress * 100.0) << "% ";

    std::cout << message << " \r";

    std::cout.flush();
  }

  void update(const int step, const int total) const
  {
    const float progress = (float)(step+1) / total;

    std::cout << "[";

    const int pos = barWidth_ * progress;

    for (int k = 0; k < barWidth_; ++k)
    {
      if (k < pos)
      {
        std::cout << "=";
      }
      else if (k == pos)
      {
        std::cout << ">";
      }
      else
      {
        std::cout << " ";
      }
    }

    std::cout << "] "
              << int(progress * 100.0) << "% ";

    std::cout << step << "/" << total << " \r";

    std::cout.flush();
  }

  void close()
  {
    std::cout << std::endl;
  }

  // Destructor

  ~progressBar() = default;
};


} // namespace

#endif
