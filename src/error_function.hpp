#pragma once

#include "matrix.hpp"

class ErrorFunction {
public:
  virtual TFloat apply(const Matrix<TFloat> &got,
                       const Matrix<TFloat> &expected) const = 0;
  virtual Matrix<TFloat> derivative(const Matrix<TFloat> &got,
                                    const Matrix<TFloat> &expected) const = 0;
};
