#pragma once

#include "../error_function.hpp"

class MeanSquaredError : public ErrorFunction {
  virtual Expression apply(const Matrix<Expression> &got,
                           const Matrix<Expression> &expected) const override;
};
