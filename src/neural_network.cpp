#include <iostream>
#include <random>

#include "neural_network.hpp"

using namespace std;

const TFloat RANDOM_PARAMETER_MEAN = 0.0;
const TFloat RANDOM_PARAMETER_STDDEV = 1.0;

void randomize_matrix(Matrix &matrix, random_device &rd,
                      normal_distribution<TFloat> &distribution) {
  for (size_t i = 0; i != matrix.data.size(); i++)
    matrix.data[i] = distribution(rd);
}

NeuralNetwork::LinearLayer::LinearLayer(
    size_t inputs, size_t outputs,
    std::optional<std::shared_ptr<ActivationFunction>> activation_function)
    : weights(outputs, inputs), bias(outputs, 1), weights_grad(outputs, inputs),
      bias_grad(outputs, 1),
      activation_function(std::move(activation_function)) {}

Matrix NeuralNetwork::LinearLayer::forward() {
  Matrix result = this->weights.dot(*this->input) + this->bias;

  if (this->activation_function.has_value())
    result = this->activation_function->get()->apply(result);

  return result;
}

void NeuralNetwork::LinearLayer::backward(const Matrix &layer_error) {
  this->weights_grad += layer_error.dot(this->input->transpose());
  this->bias_grad += layer_error;
}

Matrix
NeuralNetwork::LinearLayer::previous_layer_error(const Matrix &layer_error) {
  Matrix result = this->weights.transpose().dot(layer_error);

  if (this->activation_function.has_value())
    result *= this->activation_function->get()->derivative(*this->input);

  return result;
}

void NeuralNetwork::LinearLayer::flush_gradients(TFloat learning_rate) {
  this->weights -= this->weights_grad * learning_rate;
  this->bias -= this->bias_grad * learning_rate;

  this->weights_grad.zeroize();
  this->bias_grad.zeroize();
}

NeuralNetwork::NeuralNetwork(
    size_t inputs_count,
    vector<pair<size_t, optional<shared_ptr<ActivationFunction>>>> layers,
    shared_ptr<ErrorFunction> error_function)
    : error_function(std::move(error_function)), inputs_count(inputs_count) {
  size_t previous_layer_outputs_count = inputs_count;

  for (auto &[layer_size, layer_activation_function] : layers) {
    this->layers.push_back(LinearLayer(previous_layer_outputs_count, layer_size,
                                       std::move(layer_activation_function)));

    previous_layer_outputs_count = layer_size;
  }

  this->randomize_parameters(RANDOM_PARAMETER_MEAN, RANDOM_PARAMETER_STDDEV);
}

void NeuralNetwork::randomize_parameters(TFloat mean, TFloat stddev) {
  random_device rd;
  normal_distribution distribution(mean, stddev);

  for (auto &layer : this->layers) {
    randomize_matrix(layer.weights, rd, distribution);
    randomize_matrix(layer.bias, rd, distribution);
  }
}

Matrix NeuralNetwork::forward(const Matrix &input) {
  this->layers.at(0).input = input;

  for (size_t i = 0; i != this->layers.size() - 1; i++) {
    this->layers[i + 1].input = this->layers[i].forward();
  }

  this->output = layers.back().forward();

  return this->output;
}

TFloat NeuralNetwork::expect(const Matrix &expected_output) {
  this->expected_output = expected_output;

  return this->error_function->apply(this->output, this->expected_output);
}

void NeuralNetwork::backward(TFloat learning_rate) {
  Matrix layer_error =
      this->error_function->derivative(this->output, this->expected_output);

  for (ssize_t i = this->layers.size() - 1; i != -1; i--) {
    auto &layer = this->layers[i];

    layer.backward(layer_error);

    if (i != 0) {
      layer_error = layer.previous_layer_error(layer_error);
    }

    layer.flush_gradients(learning_rate);
  }
}
