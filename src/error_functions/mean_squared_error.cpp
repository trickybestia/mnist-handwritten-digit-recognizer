#include <cmath>

#include "mean_squared_error.hpp"

using namespace std;

TFloat MeanSquaredError::apply(const Matrix<TFloat> &got,
                               const Matrix<TFloat> &expected) const {
  if (got.rows() != expected.rows() || got.cols() != expected.cols())
    throw exception();

  TFloat result = 0.0;

  for (size_t i = 0; i != got.data.size(); i++) {
    result += pow(got.data[i] - expected.data[i], 2.0);
  }

  result /= 2.0 * got.data.size();

  return result;
}

Matrix<TFloat>
MeanSquaredError::derivative(const Matrix<TFloat> &got,
                             const Matrix<TFloat> &expected) const {
  if (got.rows() != expected.rows() || got.cols() != expected.cols())
    throw exception();

  Matrix<TFloat> result(got.rows(), got.cols());

  for (size_t i = 0; i != got.data.size(); i++) {
    result.data[i] = (got.data[i] - expected.data[i]) / got.data.size();
  }

  return result;
}
