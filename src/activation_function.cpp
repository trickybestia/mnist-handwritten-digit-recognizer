#include "activation_function.hpp"

Matrix<TFloat> ActivationFunction::apply(const Matrix<TFloat> &X) const {
  Matrix<TFloat> result(X.rows(), X.cols());

  for (size_t i = 0; i != result.data.size(); i++) {
    result.data[i] = this->apply(X.data[i]);
  }

  return result;
}

Matrix<TFloat> ActivationFunction::derivative(const Matrix<TFloat> &X) const {
  Matrix<TFloat> result(X.rows(), X.cols());

  for (size_t i = 0; i != result.data.size(); i++) {
    result.data[i] = this->derivative(X.data[i]);
  }

  return result;
}
