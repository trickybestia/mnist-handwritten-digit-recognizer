#pragma once

#include "../activation_function.hpp"

class Sigmoid : public ActivationFunction {
public:
  virtual Expression apply(Expression x) const override;
};
