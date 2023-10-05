#include <cmath>

#include "sigmoid.hpp"

Expression Sigmoid::apply(Expression x) const {
  return 1.0_expr / (1.0_expr + exp(M_E, -x));
}
