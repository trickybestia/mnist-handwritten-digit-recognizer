#pragma once

#include "matrix.hpp"

class ErrorFunction {
public:
  virtual Expression apply(const Matrix<Expression> &got,
                           const Matrix<Expression> &expected) const = 0;
};
