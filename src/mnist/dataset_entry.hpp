#pragma once

#include <cstdint>

#include "../matrix.hpp"

namespace mnist {
struct DatasetEntry {
private:
  uint8_t _label;
  Eigen::Matrix<TFloat, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _image;

public:
  DatasetEntry(
      uint8_t label,
      Eigen::Matrix<TFloat, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          image)
      : _label(label), _image(image) {}

  uint8_t label() const { return this->_label; }
  const Eigen::Matrix<TFloat, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &
  image() const {
    return this->_image;
  }
};
} // namespace mnist
