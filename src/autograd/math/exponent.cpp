#include <cmath>

#include "../autograd.hpp"

using namespace std;

class ExponentExpression : public ExpressionBase {
private:
  const TFloat base;
  const Expression power;

protected:
  virtual TFloat compute_value() override {
    return ::pow(this->base, this->power->value());
  }

  virtual Expression compute_derivative(Variable wrt) override {
    return this->shared_from_this() *
           make_shared<ConstExpression>(std::log(this->base)) *
           this->power->derivative(wrt);
  }

public:
  ExponentExpression(TFloat base, Expression power)
      : base(base), power(power) {}
};

Expression exp(TFloat base, Expression power) {
  return ExpressionBase::with_dependency(
      make_shared<ExponentExpression>(base, power), {power});
}
