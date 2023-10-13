#include "activation_function.hpp"

size_t ActivationFunction::parameters_count() const { return 0; }

void ActivationFunction::set_parameters(TFloat *) {}

void ActivationFunction::set_gradient(TFloat *) {}

Matrix ActivationFunction::forward(Matrix input) {
  this->_input = input;

  Matrix result(input.rows(), input.cols());

  for (size_t i = 0; i != input.size(); i++) {
    result(i) = this->apply(input(i));
  }

  return result;
}

void ActivationFunction::backward(const Matrix &) {}

Matrix ActivationFunction::previous_layer_error(const Matrix &layer_error) {
  Matrix result(layer_error.rows(), layer_error.cols());

  for (size_t i = 0; i != layer_error.size(); i++) {
    result(i) = layer_error(i) * this->derivative(this->_input(i));
  }

  return result;
}
