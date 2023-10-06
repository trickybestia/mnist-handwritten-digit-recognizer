#include <cmath>

#include "leaky_relu.hpp"

LeakyReLU::LeakyReLU(TFloat alpha) : factor(-alpha) {}

DifferentiableValue LeakyReLU::apply(const DifferentiableValue &x) const {
  if (x.value() > 0.0)
    return x;

  return this->factor * x;
}
