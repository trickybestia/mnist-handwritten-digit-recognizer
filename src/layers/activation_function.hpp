#pragma once

#include "../layer.hpp"
#include "../matrix.hpp"

class ActivationFunction : public Layer {
private:
  Matrix _input;

public:
  virtual TFloat apply(TFloat x) const = 0;
  virtual TFloat derivative(TFloat x) const = 0;

  virtual size_t parameters_count() const override;

  virtual void set_parameters(TFloat *value) override;
  virtual void set_gradient(TFloat *value) override;

  virtual Matrix forward(Matrix input) override;
  virtual void backward(const Matrix &layer_error) override;

  virtual Matrix previous_layer_error(const Matrix &layer_error) override;
};
