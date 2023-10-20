#include <random>

#include "neural_network.hpp"

using namespace std;

const TFloat RANDOM_PARAMETER_MIN = -0.3;
const TFloat RANDOM_PARAMETER_MAX = 0.3;

void randomize_vector(Vector &matrix, random_device &rd,
                      uniform_real_distribution<TFloat> &distribution) {
  for (ssize_t i = 0; i != matrix.size(); i++) {
    matrix(i) = distribution(rd);
  }
}

void NeuralNetwork::randomize_parameters(TFloat min, TFloat max) {
  random_device rd;
  uniform_real_distribution distribution(min, max);

  randomize_vector(this->_parameters, rd, distribution);
}

NeuralNetwork::NeuralNetwork(vector<shared_ptr<Layer>> layers,
                             shared_ptr<ErrorFunction> error_function)
    : _layers(layers), _error_function(error_function) {
  size_t parameters_count = 0;

  for (size_t i = 0; i != this->_layers.size(); i++) {
    parameters_count += this->_layers[i]->parameters_count();
  }

  this->_parameters = Vector(parameters_count);
  this->_gradient = Vector::Zero(parameters_count);

  this->randomize_parameters(RANDOM_PARAMETER_MIN, RANDOM_PARAMETER_MAX);

  size_t parameters_offset = 0;

  for (size_t i = 0; i != this->_layers.size(); i++) {
    this->_layers[i]->set_parameters(this->_parameters.data() +
                                     parameters_offset);
    this->_layers[i]->set_gradient(this->_gradient.data() + parameters_offset);

    parameters_offset += this->_layers[i]->parameters_count();
  }
}

Vector NeuralNetwork::forward(const Vector &input) {
  this->_output = input;

  for (size_t i = 0; i != this->_layers.size(); i++) {
    this->_output = this->_layers[i]->forward(this->_output);
  }

  return this->_output;
}

TFloat NeuralNetwork::expect(const Vector &expected_output) {
  this->_expected_output = expected_output;

  return this->_error_function->apply(this->_output, this->_expected_output);
}

void NeuralNetwork::backward() {
  Vector layer_error =
      this->_error_function->derivative(this->_output, this->_expected_output);

  for (size_t i = this->_layers.size() - 1; i != 0; i--) {
    auto &layer = this->_layers[i];

    layer->backward(layer_error);

    layer_error = layer->previous_layer_error(layer_error);
  }

  this->_layers.front()->backward(layer_error);
}

Vector &NeuralNetwork::parameters() { return this->_parameters; }

Vector &NeuralNetwork::gradient() { return this->_gradient; }
