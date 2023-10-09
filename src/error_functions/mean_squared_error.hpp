#pragma once

#include "../error_function.hpp"

class MeanSquaredError : public ErrorFunction {
  virtual TFloat apply(const Matrix &got, const Matrix &expected) const;
  virtual Matrix derivative(const Matrix &got, const Matrix &expected) const;
};
