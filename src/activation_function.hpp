#pragma once

#include "float.hpp"
#include "matrix.hpp"

class ActivationFunction {
public:
  virtual TFloat apply(TFloat parameter) const = 0;
  virtual TFloat derivative(TFloat parameter) const = 0;

  Matrix apply(const Matrix &parameters) const;
  Matrix derivative(const Matrix &parameters) const;
};
