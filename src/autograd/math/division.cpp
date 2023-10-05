#include "../autograd.hpp"

using namespace std;

class DivisionExpression : public ExpressionBase {
private:
  const Expression left, right;

protected:
  virtual TFloat compute_value() override {
    return this->left->value() / this->right->value();
  }

  virtual Expression compute_derivative(Expression wrt) override {
    return (this->left->derivative(wrt) * this->right -
            this->left * this->right->derivative(wrt)) /
           (this->right * this->right);
  }

public:
  DivisionExpression(Expression left, Expression right)
      : left(left), right(right) {}
};

Expression operator/(Expression left, Expression right) {
  return ExpressionBase::with_dependency(
      make_shared<DivisionExpression>(left, right), {left, right});
}
