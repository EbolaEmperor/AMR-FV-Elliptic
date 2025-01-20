#pragma once

template <typename T>
class BaseIterator {
protected:
  T &object_;

  unsigned current_;

public:
  BaseIterator(T &obj) : object_(obj), current_(0) {}

  virtual bool ok() const { return current_ < object_.size(); }

  virtual BaseIterator<T> &operator++() {
    ++current_;
    return *this;
  }

  void setIndex(unsigned idx) { current_ = idx; }

  unsigned index() const { return current_; }

  bool operator==(const BaseIterator<T> &rhs) const {
    return index() == rhs.index();
  }
};