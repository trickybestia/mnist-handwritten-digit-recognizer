#include "autograd.hpp"
#include <cmath>

using namespace std;

shared_ptr<ValueExpression> operator""_expr(long double value) {
  return make_shared<ValueExpression>(value);
}

void ExpressionBase::reset_cache() {
  if (this->_value.has_value()) {
    this->_value = {};

    for (weak_ptr<ExpressionBase> weak_dependency : this->dependencies)
      if (shared_ptr<ExpressionBase> dependency = weak_dependency.lock())
        dependency->reset_cache();

    for (auto [weak_derivative, _] : this->derivatives)
      if (shared_ptr<ExpressionBase> derivative = weak_derivative.lock())
        derivative->reset_cache();
  }
}

TFloat ExpressionBase::value() {
  if (!this->_value.has_value())
    this->_value = this->compute_value();

  return *this->_value;
}

Expression ExpressionBase::derivative(Expression wrt) {
  if (!this->derivatives.contains(wrt))
    this->derivatives[wrt] = this->compute_derivative(wrt);

  return this->derivatives[wrt];
}

Expression ExpressionBase::with_dependency(
    Expression expression, std::initializer_list<Expression> prerequisites) {
  for (Expression prerequisite : prerequisites)
    prerequisite->dependencies.insert(expression);

  return expression;
}

ValueExpression::ValueExpression(TFloat value) : _value(value) {}

void ValueExpression::set_value(TFloat value) {
  this->_value = value;

  this->reset_cache();
}

TFloat ValueExpression::compute_value() { return this->_value; }

Expression ValueExpression::compute_derivative(Expression wrt) {
  if (wrt.get() == this)
    return 1.0_expr;

  return 0.0_expr;
}
