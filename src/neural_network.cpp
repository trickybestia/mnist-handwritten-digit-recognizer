#include <random>

#include "neural_network.hpp"

using namespace std;

NeuralNetwork::NeuralNetwork(size_t inputs_count, size_t hidden_layer_size,
                             size_t outputs_count,
                             unique_ptr<ActivationFunction> activation_function,
                             unique_ptr<ErrorFunction> error_function)
    : W1(inputs_count, hidden_layer_size), W2(hidden_layer_size, outputs_count),
      B1(1, hidden_layer_size), B2(1, outputs_count),
      activation_function(std::move(activation_function)),
      error_function(std::move(error_function)), inputs_count(inputs_count),
      hidden_layer_size(hidden_layer_size), outputs_count(outputs_count) {
  this->randomize();
}

void NeuralNetwork::randomize() {
  const TFloat MIN = -2.0;
  const TFloat MAX = 2.0;

  this->W1.randomize(MIN, MAX);
  this->W2.randomize(MIN, MAX);

  this->B1.randomize(MIN, MAX);
  this->B2.randomize(MIN, MAX);
}

const Matrix &NeuralNetwork::forward(const Matrix &input) {
  this->A0 = input;

  this->Z1 = this->A0.dot(this->W1) + this->B1;
  this->A1 = this->activation_function->apply(this->Z1);
  this->Z2 = this->A1.dot(this->W2) + this->B2;

  return this->Z2;
}

void NeuralNetwork::backward(const Matrix &expected_output,
                             TFloat learning_rate) {
  Matrix output_layer_error =
      this->error_function->derivative(this->Z2, expected_output);
  Matrix hidden_layer_error = output_layer_error.dot(this->W2.transpose()) *
                              this->activation_function->derivative(this->Z1);

  Matrix grad_W2 = this->A1.transpose().dot(output_layer_error);
  Matrix grad_W1 = this->A0.transpose().dot(hidden_layer_error);

  this->W1 = this->W1 - grad_W1 * learning_rate;
  this->W2 = this->W2 - grad_W2 * learning_rate;

  Matrix &grad_B1 = hidden_layer_error;
  Matrix &grad_B2 = output_layer_error;

  this->B1 = this->B1 - grad_B1 * learning_rate;
  this->B2 = this->B2 - grad_B2 * learning_rate;
}
