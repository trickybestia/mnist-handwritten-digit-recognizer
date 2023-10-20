#include <cmath>

#include "mean.hpp"

using namespace std;

TFloat Mean::apply(const Vector &got, const Vector &expected) {
  return (got - expected).cwiseAbs().mean();
}

Vector Mean::derivative(const Vector &got, const Vector &expected) {
  if (got.rows() != expected.rows() || got.cols() != expected.cols())
    throw exception();

  Vector result(got.size());

  for (ssize_t i = 0; i != got.size(); i++) {
    if (got(i) >= expected(i)) {
      result(i) = 1.0;
    } else {
      result(i) = -1.0;
    }
  }

  result /= got.size();

  return result;
}
