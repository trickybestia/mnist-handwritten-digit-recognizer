#pragma once

#include "../error_function.hpp"

class MeanSquaredError : public ErrorFunction {
  virtual TFloat apply(const Matrix<TFloat> &got,
                       const Matrix<TFloat> &expected) const;
  virtual Matrix<TFloat> derivative(const Matrix<TFloat> &got,
                                    const Matrix<TFloat> &expected) const;
};
