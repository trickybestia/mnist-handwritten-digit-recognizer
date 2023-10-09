#include "activation_function.hpp"

Matrix ActivationFunction::apply(const Matrix &X) const {
  Matrix result(X.rows(), X.cols());

  for (size_t i = 0; i != result.data.size(); i++) {
    result.data[i] = this->apply(X.data[i]);
  }

  return result;
}

Matrix ActivationFunction::derivative(const Matrix &X) const {
  Matrix result(X.rows(), X.cols());

  for (size_t i = 0; i != result.data.size(); i++) {
    result.data[i] = this->derivative(X.data[i]);
  }

  return result;
}
