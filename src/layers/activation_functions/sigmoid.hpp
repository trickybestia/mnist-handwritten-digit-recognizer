#pragma once

#include "../activation_function.hpp"

class Sigmoid : public ActivationFunction {
private:
  Vector _output;

public:
  virtual Vector forward(const Vector &input) override;

  virtual Vector previous_layer_error(const Vector &layer_error) override;
};
