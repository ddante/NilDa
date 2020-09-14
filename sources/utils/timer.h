#ifndef TIMER_H
#define TIMER_H

#include <ctime>
#include <chrono>

namespace NilDa
{


class timer
{
private:

  std::chrono::system_clock::time_point startTime_, stopTime_, savedTime_;

public:

  // Constructor

  timer():
    startTime_{std::chrono::system_clock::now()},
    stopTime_{0},
    savedTime_{0}
  {}

  // Member functions

  void pause()
  {
    savedTime_ = std::chrono::system_clock::now();
  }

  void restart()
  {
    assert(savedTime_);

    stopTime_ += std::chrono::system_clock::now() - savedTime_;
  }

  template <class T>
  T elapsedTime()
  {
    auto endTime = std::chrono::system_clock::now;

    return
    std::chrono::duration_cast<T>(endTime - startTime_ - stopTime_).count();
  }


  // Destructor

  ~timer();

};


} // namespace

#endif
