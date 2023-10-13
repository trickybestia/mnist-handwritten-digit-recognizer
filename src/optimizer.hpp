#pragma once

#include "function.hpp"

class Optimizer {
protected:
  Function &function;

  Optimizer(Function &function) : function(function) {}

public:
  virtual void step() = 0;
};
