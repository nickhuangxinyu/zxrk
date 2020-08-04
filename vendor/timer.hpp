#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <unordered_map>
#include <time.h>
#include <libconfig.h++>
#include <sys/time.h>

#include <cmath>
#include <vector>
#include <iostream>
#include <string>

class Timer {
 public:
  Timer() = default;
  ~Timer() = default;

  void StartTimer() {
    timeval current_time;
    gettimeofday(&current_time, NULL);
    timer_sec = current_time.tv_sec;
    timer_usec = current_time.tv_usec;
    is_timer_on = true;
  }

  void EndTimer(const std::string & label="") {
    if (!is_timer_on) {
      printf("timer is not on, cant end!\n");
      return;
    }
    timeval current_time;
    gettimeofday(&current_time, NULL);
    printf("[%s]timer running time: %ld %ld\n", label.c_str(), current_time.tv_sec - timer_sec, current_time.tv_usec-timer_usec);
    is_timer_on = false;
  }

 private:
  long int timer_sec;
  long int timer_usec;
  bool is_timer_on;
};

#endif  // TIMER_HPP_
