#include "leaky_relu.hpp"

LeakyReLU::LeakyReLU(TFloat factor) : _factor(factor) {}

Vector LeakyReLU::forward(const Vector &input) {
  Vector result(input.size());

  for (ssize_t i = 0; i != input.size(); i++) {
    TFloat x = input(i);

    if (x > 0.0) {
      result(i) = x;
    } else {
      result(i) = x * this->_factor;
    }
  }

  return result;
}

Vector LeakyReLU::previous_layer_error(const Vector &previous_layer_error) {
  Vector result(previous_layer_error.size());

  for (ssize_t i = 0; i != previous_layer_error.size(); i++) {
    TFloat x = previous_layer_error(i);

    if (x > 0.0) {
      result(i) = 1.0;
    } else {
      result(i) = this->_factor;
    }
  }

  return result;
}
