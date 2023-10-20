#pragma once

#include "../error_function.hpp"
#include "../layers/activation_function.hpp"

class CrossEntropySoftmax : public ErrorFunction, public ActivationFunction {
private:
  Vector _expected_output, _output;

public:
  virtual TFloat apply(const Vector &got, const Vector &expected) override;
  virtual Vector derivative(const Vector &got, const Vector &expected) override;

  virtual Vector forward(const Vector &input) override;

  virtual Vector previous_layer_error(const Vector &layer_error) override;
};
