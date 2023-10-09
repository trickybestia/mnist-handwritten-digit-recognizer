#pragma once

#include "../activation_function.hpp"

class Tanh : public ActivationFunction {
public:
  virtual TFloat apply(TFloat x) const override;
  virtual TFloat derivative(TFloat x) const override;
};
