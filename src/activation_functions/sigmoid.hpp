#pragma once

#include "../activation_function.hpp"

class Sigmoid : public ActivationFunction {
public:
  virtual DifferentiableValue
  apply(const DifferentiableValue &x) const override;
};
