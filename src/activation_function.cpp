#include "activation_function.hpp"

Matrix<Expression>
ActivationFunction::apply(const Matrix<Expression> &X) const {
  Matrix<Expression> result(X.rows(), X.cols());

  for (size_t i = 0; i != result.data.size(); i++) {
    result.data[i] = this->apply(X.data[i]);
  }

  return result;
}
