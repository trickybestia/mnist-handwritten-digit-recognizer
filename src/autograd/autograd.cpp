#include <cmath>

#include "autograd.hpp"

using namespace std;

shared_ptr<ConstExpression> operator""_expr(long double value) {
  return make_shared<ConstExpression>(value);
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

Expression ExpressionBase::derivative(Variable wrt) {
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

ConstExpression::ConstExpression(TFloat value) : _value(value) {}

TFloat ConstExpression::compute_value() { return this->_value; }

Expression ConstExpression::compute_derivative(Variable) { return 0.0_expr; }

VariableExpression::VariableExpression(TFloat value) : _value(value) {}

void VariableExpression::set_value(TFloat value) {
  this->_value = value;

  this->reset_cache();
}

TFloat VariableExpression::compute_value() { return this->_value; }

Expression VariableExpression::compute_derivative(Variable wrt) {
  if (wrt.get() == this)
    return 1.0_expr;

  return 0.0_expr;
}
