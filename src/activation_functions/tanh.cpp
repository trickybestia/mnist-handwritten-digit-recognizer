#include <cmath>

#include "tanh.hpp"

DifferentiableValue Tanh::apply(const DifferentiableValue &x) const {
  DifferentiableValue exponent = pow(M_E, 2.0_diff * x);

  DifferentiableValue result = (exponent - 1.0_diff) / (exponent + 1.0_diff);

  return result;
}
