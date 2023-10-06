#include <cmath>

#include "sigmoid.hpp"

DifferentiableValue Sigmoid::apply(const DifferentiableValue &x) const {
  return 1.0_diff / (1.0_diff + pow(M_E, -x));
}
