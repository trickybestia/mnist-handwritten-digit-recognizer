#include "../autograd.hpp"

using namespace std;

Expression operator-(Expression expression) {
  return ExpressionBase::with_dependency(
      make_shared<ValueExpression>(-1.0) * expression, {expression});
}
