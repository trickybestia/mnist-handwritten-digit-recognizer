#include <cmath>

#include "sigmoid.hpp"

TFloat Sigmoid::apply(TFloat x) const { return 1.0 / (1.0 + exp(-x)); }

TFloat Sigmoid::derivative(TFloat x) const {
  TFloat value = this->apply(x);

  return value * (1.0 - value);
}
