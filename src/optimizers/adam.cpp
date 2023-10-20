#include <cmath>

#include "adam.hpp"

Adam::Adam(Function &function, TFloat learning_rate, TFloat beta1, TFloat beta2,
           TFloat eps)
    : Optimizer(function), _learning_rate(learning_rate), _beta1(beta1),
      _beta2(beta2), _eps(eps), _iteration(1),
      _parameters_m(Vector::Zero(function.parameters().size())),
      _parameters_v(Vector::Zero(function.parameters().size())) {}

void Adam::step() {
  Vector &parameters_gradient = this->_function.gradient();

  this->_parameters_m = this->_parameters_m * this->_beta1 +
                        parameters_gradient * (1.0 - this->_beta1);
  this->_parameters_v =
      this->_parameters_v.array() * this->_beta2 +
      parameters_gradient.array().square() * (1.0 - this->_beta2);

  Vector parameters_mt =
             this->_parameters_m / (1.0 - pow(this->_beta1, this->_iteration)),
         parameters_vt =
             this->_parameters_v / (1.0 - pow(this->_beta2, this->_iteration));

  this->_function.parameters().array() -=
      parameters_mt.array() / (parameters_vt.array().sqrt() + this->_eps) *
      this->_learning_rate;
}
