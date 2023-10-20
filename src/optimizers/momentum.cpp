#include "momentum.hpp"

Momentum::Momentum(Function &function, TFloat learning_rate, TFloat beta)
    : Optimizer(function), _learning_rate(learning_rate), _beta(beta),
      _parameters_v(Vector::Zero(function.parameters().size())) {}

void Momentum::step() {
  this->_parameters_v = this->_parameters_v * this->_beta +
                        this->_function.gradient() * this->_learning_rate;

  this->_function.parameters() -= this->_parameters_v;
}
