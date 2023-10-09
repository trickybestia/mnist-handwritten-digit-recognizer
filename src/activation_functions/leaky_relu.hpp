#pragma once

#include "../activation_function.hpp"

class LeakyReLU : public ActivationFunction {
private:
  const TFloat factor;

public:
  LeakyReLU(TFloat factor);

  virtual TFloat apply(TFloat x) const override;
  virtual TFloat derivative(TFloat x) const override;
};
