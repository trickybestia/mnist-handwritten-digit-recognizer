#pragma once

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

typedef float TFloat;

class ExpressionBase;
class VariableExpression;
class ConstExpression;

typedef std::shared_ptr<ExpressionBase> Expression;
typedef std::shared_ptr<VariableExpression> Variable;

std::shared_ptr<ConstExpression> operator""_expr(long double);

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
  std::map<std::weak_ptr<VariableExpression>, Expression, std::owner_less<>>
      derivatives;
  std::set<std::weak_ptr<ExpressionBase>, std::owner_less<>> dependencies;

protected:
  virtual TFloat compute_value() = 0;
  virtual Expression compute_derivative(Variable wrt) = 0;

  void reset_cache();

public:
  TFloat value();
  Expression derivative(Variable wrt);

  static Expression
  with_dependency(Expression expression,
                  std::initializer_list<Expression> prerequisites);
};

class ConstExpression : public ExpressionBase {
private:
  const TFloat _value;

protected:
  virtual TFloat compute_value() override;
  virtual Expression compute_derivative(Variable wrt) override;

public:
  ConstExpression(TFloat value);
};

class VariableExpression : public ExpressionBase {
private:
  TFloat _value;

protected:
  virtual TFloat compute_value() override;
  virtual Expression compute_derivative(Variable wrt) override;

public:
  VariableExpression(TFloat value);

  void set_value(TFloat value);
};
