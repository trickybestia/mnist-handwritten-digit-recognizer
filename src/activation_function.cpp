#include "activation_function.hpp"

Matrix ActivationFunction::apply(const Matrix &parameters) const {
  Matrix result(parameters.rows(), parameters.cols());

  for (size_t i = 0; i != result.data.size(); i++) {
    result.data[i] = this->apply(parameters.data[i]);
  }

  return result;
}

Matrix ActivationFunction::derivative(const Matrix &parameters) const {
  Matrix result(parameters.rows(), parameters.cols());

  for (size_t i = 0; i != result.data.size(); i++) {
    result.data[i] = this->derivative(parameters.data[i]);
  }

  return result;
}
