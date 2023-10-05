#include "../autograd.hpp"

using namespace std;

class AdditionExpression : public ExpressionBase {
private:
  const Expression left, right;

protected:
  virtual TFloat compute_value() override {
    return this->left->value() + this->right->value();
  }

  virtual Expression compute_derivative(Variable wrt) override {
    return this->left->derivative(wrt) + this->right->derivative(wrt);
  }

public:
  AdditionExpression(Expression left, Expression right)
      : left(left), right(right) {}
};

Expression operator+(Expression left, Expression right) {
  return ExpressionBase::with_dependency(
      make_shared<AdditionExpression>(left, right), {left, right});
}