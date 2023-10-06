#include <random>

#include "neural_network.hpp"

using namespace std;

const TFloat RANDOM_PARAMETER_MEAN = 0.0;
const TFloat RANDOM_PARAMETER_STDDEV = 1.0;

pair<Matrix<DifferentiableValue>, VariableId>
randomize_variable_matrix(size_t rows, size_t cols,
                          VariableId &variable_id_offset) {
  random_device rd;
  normal_distribution<TFloat> distribution(RANDOM_PARAMETER_MEAN,
                                           RANDOM_PARAMETER_STDDEV);

  pair<Matrix<DifferentiableValue>, VariableId> result{
      Matrix<DifferentiableValue>(rows, cols), variable_id_offset};

  for (size_t i = 0; i != result.first.data.size(); i++)
    result.first.data[i] =
        DifferentiableValue(distribution(rd), variable_id_offset + i);

  variable_id_offset += result.first.data.size();

  return result;
}

Matrix<DifferentiableValue> grad(const DifferentiableValue &value, size_t rows,
                                 size_t cols, VariableId variable_id_offset) {
  Matrix<DifferentiableValue> result(rows, cols);

  for (size_t i = 0; i != rows * cols; i++)
    result.data[i] =
        DifferentiableValue(value.derivative(variable_id_offset + i));

  return result;
}

Matrix<DifferentiableValue> create_const_matrix(const Matrix<TFloat> &values) {
  Matrix<DifferentiableValue> result(values.rows(), values.cols());

  for (size_t i = 0; i != values.data.size(); i++) {
    result.data[i] = DifferentiableValue(values.data[i]);
  }

  return result;
}

Matrix<DifferentiableValue>
create_variable_matrix(const Matrix<TFloat> &values,
                       VariableId &variable_id_offset) {
  Matrix<DifferentiableValue> result(values.rows(), values.cols());

  for (size_t i = 0; i != values.data.size(); i++) {
    result.data[i] =
        DifferentiableValue(values.data[i], variable_id_offset + i);
  }

  variable_id_offset += values.data.size();

  return result;
}

NeuralNetwork::NeuralNetwork(size_t inputs_count, size_t hidden_layer_size,
                             size_t outputs_count,
                             unique_ptr<ActivationFunction> activation_function,
                             unique_ptr<ErrorFunction> error_function)
    : activation_function(std::move(activation_function)),
      error_function(std::move(error_function)), inputs_count(inputs_count),
      hidden_layer_size(hidden_layer_size), outputs_count(outputs_count) {
  VariableId variable_id_offset = 0;

  this->W1 = randomize_variable_matrix(hidden_layer_size, inputs_count,
                                       variable_id_offset);
  this->W2 = randomize_variable_matrix(outputs_count, hidden_layer_size,
                                       variable_id_offset);
  this->B1 =
      randomize_variable_matrix(hidden_layer_size, 1, variable_id_offset);
  this->B2 = randomize_variable_matrix(outputs_count, 1, variable_id_offset);
}

Matrix<TFloat> NeuralNetwork::forward(const Matrix<TFloat> &input) {
  this->input = create_const_matrix(input);

  auto Z1 = this->W1.first.dot(this->input) + this->B1.first;
  auto A1 = this->activation_function->apply(Z1);
  this->output = this->W2.first.dot(A1) + this->B2.first;

  Matrix<TFloat> result(this->outputs_count, 1);

  for (size_t i = 0; i != this->outputs_count; i++)
    result.data[i] = this->output.data[i].value();

  return result;
}

TFloat NeuralNetwork::expect(const Matrix<TFloat> &expected_output) {
  this->error = this->error_function->apply(
      this->output, create_const_matrix(expected_output));

  return this->error.value();
}

void NeuralNetwork::backward(TFloat learning_rate) {
  DifferentiableValue _learning_rate = DifferentiableValue(learning_rate);

  Matrix<DifferentiableValue> grad_W1 =
                                  grad(error, this->W1.first.rows(),
                                       this->W1.first.cols(), this->W1.second),
                              grad_W2 =
                                  grad(error, this->W2.first.rows(),
                                       this->W2.first.cols(), this->W2.second),
                              grad_B1 =
                                  grad(error, this->B1.first.rows(),
                                       this->B1.first.cols(), this->B1.second),
                              grad_B2 =
                                  grad(error, this->B2.first.rows(),
                                       this->B2.first.cols(), this->B2.second);

  this->W1.first -= grad_W1 * _learning_rate;
  this->W2.first -= grad_W2 * _learning_rate;
  this->B1.first -= grad_B1 * _learning_rate;
  this->B2.first -= grad_B2 * _learning_rate;
}
