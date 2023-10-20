#pragma once

#include "../error_function.hpp"

class Mean : public ErrorFunction {
  virtual TFloat apply(const Vector &got, const Vector &expected) override;
  virtual Vector derivative(const Vector &got, const Vector &expected) override;
};
