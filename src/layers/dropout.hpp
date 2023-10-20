#pragma once

#include <random>

#include "../layer.hpp"

class Dropout : public Layer {
private:
  Vector _factors;
  std::mt19937_64 _rng;
  std::bernoulli_distribution _distribution;

public:
  Dropout(float probability);

  virtual size_t parameters_count() const override;

  virtual void set_parameters(TFloat *value) override;
  virtual void set_gradient(TFloat *value) override;

  virtual Vector forward(const Vector &input) override;

  virtual void backward(const Vector &layer_error) override;

  virtual Vector previous_layer_error(const Vector &layer_error) override;
};
