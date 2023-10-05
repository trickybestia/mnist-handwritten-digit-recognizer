#pragma once

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

typedef float TFloat;

class ExpressionBase;
class ValueExpression;

typedef std::shared_ptr<ExpressionBase> Expression;

std::shared_ptr<ValueExpression> operator""_expr(long double);

Expression operator-(Expression expression);

Expression operator+(Expression left, Expression right);
Expression operator-(Expression left, Expression right);
Expression operator*(Expression left, Expression right);
Expression operator/(Expression left, Expression right);

Expression pow(Expression base, TFloat power);
Expression log(TFloat base, Expression x);
Expression exp(TFloat base, Expression power);

class ExpressionBase : public std::enable_shared_from_this<ExpressionBase> {
private:
  std::optional<TFloat> _value;
  std::map<std::weak_ptr<ExpressionBase>, Expression, std::owner_less<>>
      derivatives;
  std::set<std::weak_ptr<ExpressionBase>, std::owner_less<>> dependencies;

protected:
  virtual TFloat compute_value() = 0;
  virtual Expression compute_derivative(Expression wrt) = 0;

  void reset_cache();

public:
  TFloat value();
  Expression derivative(Expression wrt);

  static Expression
  with_dependency(Expression expression,
                  std::initializer_list<Expression> prerequisites);
};

class ValueExpression : public ExpressionBase {
private:
  TFloat _value;

protected:
  virtual TFloat compute_value() override;
  virtual Expression compute_derivative(Expression wrt) override;

public:
  ValueExpression(TFloat value);

  void set_value(TFloat value);
};
