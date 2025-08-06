#ifndef SERF_POST_OFFICE_RESULT_H
#define SERF_POST_OFFICE_RESULT_H

#include "array.h"

class PostOfficeResult {
 public:
  PostOfficeResult(const Array<int> &office_positions, int total_app_cost) :
      office_positions_(office_positions),
      total_app_cost_(total_app_cost) {}

  Array<int> office_positions() {
    return office_positions_;
  }

  int total_app_cost() const {
    return total_app_cost_;
  }

 private:
  Array<int> office_positions_;
  int total_app_cost_;
};

#endif  // SERF_POST_OFFICE_RESULT_H
