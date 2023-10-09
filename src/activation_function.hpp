#pragma once

#include "matrix.hpp"

class ActivationFunction {
public:
  virtual ~ActivationFunction() {}

  virtual TFloat apply(TFloat x) const = 0;
  virtual TFloat derivative(TFloat x) const = 0;

  Matrix<TFloat> apply(const Matrix<TFloat> &X) const;
  Matrix<TFloat> derivative(const Matrix<TFloat> &X) const;
};
