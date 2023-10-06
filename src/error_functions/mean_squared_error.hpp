#pragma once

#include "../error_function.hpp"

class MeanSquaredError : public ErrorFunction {
  virtual DifferentiableValue
  apply(const Matrix<DifferentiableValue> &got,
        const Matrix<DifferentiableValue> &expected) const override;
};
