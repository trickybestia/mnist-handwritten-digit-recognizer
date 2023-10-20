#include "linear.hpp"

Linear::Linear(size_t inputs, size_t outputs)
    : _inputs(inputs), _outputs(outputs), _bias(nullptr, 0),
      _bias_gradient(nullptr, 0), _weights(nullptr, 0, 0),
      _weights_gradient(nullptr, 0, 0) {}

size_t Linear::parameters_count() const {
  return (this->_inputs + 1) * this->_outputs;
}

void Linear::set_parameters(TFloat *value) {
  new (&this->_weights)
      Eigen::Map<Matrix>(value, this->_outputs, this->_inputs);
  new (&this->_bias) Eigen::Map<Vector>(value + this->_inputs * this->_outputs,
                                        this->_outputs);
}

void Linear::set_gradient(TFloat *value) {
  new (&this->_weights_gradient)
      Eigen::Map<Matrix>(value, this->_outputs, this->_inputs);
  new (&this->_bias_gradient) Eigen::Map<Vector>(
      value + this->_inputs * this->_outputs, this->_outputs);
}

Vector Linear::forward(const Vector &input) {
  this->_input = input;

  return this->_weights * input + this->_bias;
}

void Linear::backward(const Vector &layer_error) {
  this->_weights_gradient += layer_error * this->_input.transpose();
  this->_bias_gradient += layer_error;
}

Vector Linear::previous_layer_error(const Vector &layer_error) {
  return this->_weights.transpose() * layer_error;
}
