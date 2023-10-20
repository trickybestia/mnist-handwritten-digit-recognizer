#include <cmath>

#include "sigmoid.hpp"

Vector Sigmoid::forward(const Vector &input) {
  this->_output = 1.0 / (1.0 + (-input).array().exp());

  return this->_output;
}

Vector Sigmoid::previous_layer_error(const Vector &layer_error) {
  return layer_error.array() * this->_output.array() *
         (1.0 - this->_output.array());
}
