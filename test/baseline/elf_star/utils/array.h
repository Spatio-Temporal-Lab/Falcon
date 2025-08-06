#ifndef SERF_ARRAY_H
#define SERF_ARRAY_H

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <vector>
#include <iostream>

template<typename T>
class Array {
 public:
  Array<T>() = default;

  explicit Array<T>(int length) : length_(length) {
    data_ = std::make_unique<T[]>(length_);
  }

  explicit Array<T>(const std::vector<T> &vec) : length_(vec.size()) {
    data_ = std::make_unique<T[]>(length_);
    std::copy(vec.begin(), vec.end(), begin());
  }

  Array<T>(std::initializer_list<T> list) : length_(list.size()) {
    data_ = std::make_unique<T[]>(length_);
    std::copy(list.begin(), list.end(), begin());
  }

  Array<T>(const Array<T> &other) : length_(other.length_) {
    data_ = std::make_unique<T[]>(length_);
    std::copy(other.begin(), other.end(), begin());
  }

  Array<T> &operator=(const Array<T> &right) {
    length_ = right.length_;
    data_ = std::make_unique<T[]>(right.length_);
    std::copy(right.begin(), right.end(), begin());
    return *this;
  }

  T &operator[](int index) const {
    return data_[index];
  }

  T *begin() const {
    return data_.get();
  }

  T *end() const {
    return data_.get() + length_;
  }

  int length() const {
    return length_;
  }

  // NOTE: new added
  void show() const { 
    for (int i = 0; i < length_;i++){
      std::cout << i << " : " << data_[i] << std::endl;
    }
    std::cout << "over" << std::endl;
  }

 private:
  int length_ = 0;
  std::unique_ptr<T[]> data_ = nullptr;
};

#endif  // SERF_ARRAY_H
