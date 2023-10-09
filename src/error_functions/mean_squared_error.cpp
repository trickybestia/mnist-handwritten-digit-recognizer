#include <cmath>

#include "mean_squared_error.hpp"

using namespace std;

TFloat MeanSquaredError::apply(const Matrix &got,
                               const Matrix &expected) const {
  if (got.rows() != expected.rows() || got.cols() != expected.cols())
    throw exception();

  TFloat result = 0.0;

  for (size_t i = 0; i != got.data.size(); i++) {
    result += pow(got.data[i] - expected.data[i], 2.0);
  }

  result /= got.data.size();

  return result;
}

Matrix MeanSquaredError::derivative(const Matrix &got,
                                    const Matrix &expected) const {
  if (got.rows() != expected.rows() || got.cols() != expected.cols())
    throw exception();

  Matrix result(got.rows(), got.cols());

  for (size_t i = 0; i != got.data.size(); i++) {
    result.data[i] = 2.0 * (got.data[i] - expected.data[i]) / got.data.size();
  }

  return result;
}
