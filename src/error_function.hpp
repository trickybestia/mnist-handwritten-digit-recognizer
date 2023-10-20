#pragma once

#include "matrix.hpp"

class ErrorFunction {
public:
  virtual TFloat apply(const Vector &got, const Vector &expected) = 0;
  virtual Vector derivative(const Vector &got, const Vector &expected) = 0;
};
