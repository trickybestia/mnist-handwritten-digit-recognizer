#pragma once

#include "differentiable_value.hpp"
#include "matrix.hpp"

class ActivationFunction {
public:
  virtual DifferentiableValue apply(const DifferentiableValue &x) const = 0;

  Matrix<DifferentiableValue> apply(const Matrix<DifferentiableValue> &X) const;
};
