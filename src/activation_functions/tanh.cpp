#include <cmath>

#include "tanh.hpp"

TFloat Tanh::apply(TFloat x) const {
  TFloat exponent = pow(M_E, 2.0 * x);

  TFloat result = (exponent - 1.0) / (exponent + 1.0);

  return result;
}

TFloat Tanh::derivative(TFloat x) const { return 1.0 - pow(this->apply(x), 2); }
