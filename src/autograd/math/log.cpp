#include <cmath>

#include "../autograd.hpp"

using namespace std;

class LogExpression : public ExpressionBase {
private:
  const TFloat base;
  const Expression x;

protected:
  virtual TFloat compute_value() override {
    return std::log(this->x->value()) / ::log(this->base);
  }

  virtual Expression compute_derivative(Variable wrt) override {
    return this->x->derivative(wrt) / (this->x * log(this->base, this->x));
  }

public:
  LogExpression(TFloat base, Expression x) : base(base), x(x) {}
};

Expression log(TFloat base, Expression x) {
  return ExpressionBase::with_dependency(make_shared<LogExpression>(base, x),
                                         {x});
}
