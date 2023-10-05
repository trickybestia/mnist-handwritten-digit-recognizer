#pragma once

#include "matrix.hpp"

class ActivationFunction {
public:
  virtual Expression apply(Expression x) const = 0;

  Matrix<Expression> apply(const Matrix<Expression> &X) const;
};
