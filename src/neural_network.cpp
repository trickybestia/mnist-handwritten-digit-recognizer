#include <random>

#include "neural_network.hpp"

using namespace std;

const TFloat RANDOM_PARAMETER_MEAN = 0.0;
const TFloat RANDOM_PARAMETER_STDDEV = 0.125;

Matrix<TFloat> randomize_matrix(size_t rows, size_t cols) {
  random_device rd;
  normal_distribution<TFloat> distribution(RANDOM_PARAMETER_MEAN,
                                           RANDOM_PARAMETER_STDDEV);

  Matrix<TFloat> result(rows, cols);

  for (size_t i = 0; i != result.data.size(); i++)
    result.data[i] = distribution(rd);

  return result;
}

NeuralNetwork::NeuralNetwork(size_t inputs_count, size_t hidden_layer_size,
                             size_t outputs_count,
                             unique_ptr<ActivationFunction> activation_function,
                             unique_ptr<ErrorFunction> error_function)
    : W1(randomize_matrix(hidden_layer_size, inputs_count)),
      W2(randomize_matrix(outputs_count, hidden_layer_size)),
      B1(randomize_matrix(hidden_layer_size, 1)),
      B2(randomize_matrix(outputs_count, 1)),
      activation_function(std::move(activation_function)),
      error_function(std::move(error_function)), inputs_count(inputs_count),
      hidden_layer_size(hidden_layer_size), outputs_count(outputs_count) {}

Matrix<TFloat> NeuralNetwork::forward(const Matrix<TFloat> &input) {
  this->A0 = input;

  this->Z1 = this->W1.dot(this->A0) + this->B1;
  this->A1 = this->activation_function->apply(this->Z1);
  this->Z2 = this->W2.dot(this->A1) + this->B2;

  return this->Z2;
}

TFloat NeuralNetwork::expect(const Matrix<TFloat> &expected_output) {
  this->expected_output = expected_output;

  return this->error_function->apply(this->Z2, this->expected_output);
}

void NeuralNetwork::backward(TFloat learning_rate) {
  Matrix<TFloat> output_layer_error =
      this->error_function->derivative(this->Z2, this->expected_output);
  Matrix<TFloat> hidden_layer_error =
      this->W2.transpose().dot(output_layer_error) *
      this->activation_function->derivative(this->Z1);

  Matrix<TFloat> grad_W1 = hidden_layer_error.dot(this->A0.transpose()),
                 grad_W2 = output_layer_error.dot(this->A1.transpose()),
                 &grad_B1 = hidden_layer_error, &grad_B2 = output_layer_error;

  this->W1 -= grad_W1 * learning_rate;
  this->W2 -= grad_W2 * learning_rate;
  this->B1 -= grad_B1 * learning_rate;
  this->B2 -= grad_B2 * learning_rate;
}
