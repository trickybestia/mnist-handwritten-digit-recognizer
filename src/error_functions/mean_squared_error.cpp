#include "mean_squared_error.hpp"

using namespace std;

Expression MeanSquaredError::apply(const Matrix<Expression> &got,
                                   const Matrix<Expression> &expected) const {
  if (got.rows() != expected.rows() || got.cols() != expected.cols())
    throw exception();

  Expression sum = pow(got.data[0] - expected.data[0], 2);

  for (size_t i = 1; i != got.data.size(); i++) {
    sum = sum + pow(got.data[i] - expected.data[i], 2);
  }

  return sum * make_shared<ConstExpression>(2.0 / got.data.size());
}
