#include "mean_squared_error.hpp"

using namespace std;

DifferentiableValue
MeanSquaredError::apply(const Matrix<DifferentiableValue> &got,
                        const Matrix<DifferentiableValue> &expected) const {
  if (got.rows() != expected.rows() || got.cols() != expected.cols())
    throw exception();

  DifferentiableValue result = pow(expected.data[0] - got.data[0], 2.0);

  for (size_t i = 1; i != got.data.size(); i++) {
    result += pow(expected.data[i] - got.data[i], 2.0);
  }

  result /= DifferentiableValue(got.data.size());

  return result;
}
