#pragma once

#include "../activation_function.hpp"

class Tanh : public ActivationFunction {
public:
  virtual DifferentiableValue
  apply(const DifferentiableValue &x) const override;
};
