#pragma once

#include "matrix.hpp"

class Function {
public:
  virtual Vector &parameters() = 0;
  virtual Vector &gradient() = 0;
};
