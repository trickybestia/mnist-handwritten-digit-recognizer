#include <cmath>

#include "tanh.hpp"

Vector Tanh::forward(const Vector &input) {
  this->_output = input.array().tanh();

  return this->_output;
}

Vector Tanh::previous_layer_error(const Vector &layer_error) {
  return layer_error.array() * (1.0 - this->_output.array().square());
}
