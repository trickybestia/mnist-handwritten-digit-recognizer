#pragma once

#include "matrix.hpp"

class Function {
public:
  virtual TFloat value() = 0;

  virtual Matrix &parameters() = 0;
  virtual Matrix &gradient() = 0;
};
