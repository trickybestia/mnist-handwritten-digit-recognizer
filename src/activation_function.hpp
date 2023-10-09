#pragma once

#include "matrix.hpp"

class ActivationFunction {
public:
  virtual ~ActivationFunction() {}

  virtual TFloat apply(TFloat x) const = 0;
  virtual TFloat derivative(TFloat x) const = 0;

  Matrix apply(const Matrix &X) const;
  Matrix derivative(const Matrix &X) const;
};
