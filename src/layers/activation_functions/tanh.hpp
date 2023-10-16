#pragma once

#include "../simple_activation_function.hpp"

class Tanh : public SimpleActivationFunction {
public:
  virtual TFloat apply(TFloat x) const override;
  virtual TFloat derivative(TFloat x) const override;
};
