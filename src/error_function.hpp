#pragma once

#include "differentiable_value.hpp"
#include "matrix.hpp"

class ErrorFunction {
public:
  virtual DifferentiableValue
  apply(const Matrix<DifferentiableValue> &got,
        const Matrix<DifferentiableValue> &expected) const = 0;
};
