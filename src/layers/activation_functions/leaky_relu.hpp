#pragma once

#include "../activation_function.hpp"

class LeakyReLU : public ActivationFunction {
private:
  const TFloat _factor;

public:
  LeakyReLU(TFloat factor);

  virtual Vector forward(const Vector &input) override;

  virtual Vector previous_layer_error(const Vector &layer_error) override;
};
