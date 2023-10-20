#pragma once

#include "matrix.hpp"

class Layer {
public:
  virtual size_t parameters_count() const = 0;

  virtual void set_parameters(TFloat *value) = 0;
  virtual void set_gradient(TFloat *value) = 0;

  virtual Vector forward(const Vector &input) = 0;
  virtual void backward(const Vector &layer_error) = 0;

  virtual Vector previous_layer_error(const Vector &layer_error) = 0;
};
