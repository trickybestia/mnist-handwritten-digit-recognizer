#include <cmath>

#include "../autograd.hpp"

using namespace std;

class PowExpression : public ExpressionBase {
private:
  const Expression base;
  const TFloat power;

protected:
  virtual TFloat compute_value() override {
    return std::pow(this->base->value(), this->power);
  }

  virtual Expression compute_derivative(Variable wrt) override {
    return make_shared<ConstExpression>(this->power) *
           pow(this->base, this->power - 1.0) * this->base->derivative(wrt);
  }

public:
  PowExpression(Expression base, TFloat power) : base(base), power(power) {}
};

Expression pow(Expression base, TFloat power) {
  return ExpressionBase::with_dependency(
      make_shared<PowExpression>(base, power), {base});
}
