#pragma once

#include "../layer.hpp"
#include "../matrix.hpp"

class Linear : public Layer {
private:
  size_t _inputs, _outputs;
  Vector _input;

  Eigen::Map<Vector> _bias, _bias_gradient;
  Eigen::Map<Matrix> _weights, _weights_gradient;

public:
  Linear(size_t inputs, size_t outputs);

  virtual size_t parameters_count() const override;

  virtual void set_parameters(TFloat *value) override;
  virtual void set_gradient(TFloat *value) override;

  virtual Vector forward(const Vector &input) override;
  virtual void backward(const Vector &layer_error) override;

  virtual Vector previous_layer_error(const Vector &layer_error) override;
};
