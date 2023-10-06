#include "activation_function.hpp"

Matrix<DifferentiableValue>
ActivationFunction::apply(const Matrix<DifferentiableValue> &X) const {
  Matrix<DifferentiableValue> result(X.rows(), X.cols());

  for (size_t i = 0; i != result.data.size(); i++) {
    result.data[i] = this->apply(X.data[i]);
  }

  return result;
}
