#pragma once

#include "../activation_function.hpp"

class LeakyReLU : public ActivationFunction {
private:
  const DifferentiableValue factor;

public:
  LeakyReLU(TFloat alpha);

  virtual DifferentiableValue
  apply(const DifferentiableValue &x) const override;
};
