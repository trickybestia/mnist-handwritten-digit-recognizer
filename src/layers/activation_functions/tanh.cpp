#include <cmath>

#include "tanh.hpp"

Matrix Tanh::forward(Matrix input) {
  this->_output = Matrix(input.rows(), input.cols());

  for (size_t i = 0; i != input.size(); i++) {
    TFloat exponent = exp(2.0 * input(i));

    this->_output(i) = (exponent - 1.0) / (exponent + 1.0);
  }

  return this->_output;
}

Matrix Tanh::previous_layer_error(const Matrix &layer_error) {
  Matrix result(layer_error.rows(), layer_error.cols());

  for (size_t i = 0; i != layer_error.size(); i++) {
    result(i) = layer_error(i) * (1.0 - pow(this->_output(i), 2));
  }

  return result;
}
