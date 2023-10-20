#include <cmath>

#include "mean_squared.hpp"

using namespace std;

TFloat MeanSquared::apply(const Vector &got, const Vector &expected) {
  return (got - expected).array().square().mean();
}

Vector MeanSquared::derivative(const Vector &got, const Vector &expected) {
  return (got - expected) * (2.0 / got.size());
}
