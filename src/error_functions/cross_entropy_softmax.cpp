#include <cmath>

#include "cross_entropy_softmax.hpp"

using namespace std;

TFloat CrossEntropySoftmax::apply(const Vector &got, const Vector &expected) {
  return -(expected.array() * got.array().log()).sum();
}

Vector CrossEntropySoftmax::derivative(const Vector &, const Vector &expected) {
  this->_expected_output = expected;

  return Vector();
}

Vector CrossEntropySoftmax::forward(const Vector &input) {
  Vector exp_input = (input.array() - input.maxCoeff()).exp();

  this->_output = exp_input / exp_input.sum();

  return this->_output;
}

Vector CrossEntropySoftmax::previous_layer_error(const Vector &) {
  return this->_output - this->_expected_output;
}
