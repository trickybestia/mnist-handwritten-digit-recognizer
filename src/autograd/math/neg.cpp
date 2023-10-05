#include "../autograd.hpp"

using namespace std;

class NegExpression : public ExpressionBase {
private:
  const Expression x;

protected:
  virtual TFloat compute_value() override { return -this->x->value(); }

  virtual Expression compute_derivative(Variable wrt) override {
    return -this->x->derivative(wrt);
  }

public:
  NegExpression(Expression x) : x(x) {}
};

Expression operator-(Expression expression) {
  return ExpressionBase::with_dependency(make_shared<NegExpression>(expression),
                                         {expression});
}
