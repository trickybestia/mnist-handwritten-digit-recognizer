#include <random>

#include "neural_network.hpp"

using namespace std;

const TFloat RANDOM_PARAMETER_MIN = -2.0;
const TFloat RANDOM_PARAMETER_MAX = 2.0;

Matrix<Expression> randomize_variable_matrix(size_t rows, size_t cols) {
  random_device rd;
  uniform_real_distribution<TFloat> distribution(RANDOM_PARAMETER_MIN,
                                                 RANDOM_PARAMETER_MAX);

  Matrix<Expression> result(rows, cols);

  for (size_t i = 0; i != result.data.size(); i++)
    result.data[i] = make_shared<VariableExpression>(distribution(rd));

  return result;
}

Matrix<Expression> create_variable_matrix(size_t rows, size_t cols) {
  Matrix<Expression> result(rows, cols);

  for (size_t i = 0; i != result.data.size(); i++)
    result.data[i] = make_shared<VariableExpression>(0.0);

  return result;
}

Matrix<TFloat> grad(Expression value, Matrix<Expression> &X) {
  Matrix<TFloat> result(X.rows(), X.cols());

  for (size_t i = 0; i != X.data.size(); i++)
    result.data[i] =
        value->derivative(dynamic_pointer_cast<VariableExpression>(X.data[i]))
            ->value();

  return result;
}

void update_value_expression_matrix(Matrix<Expression> &m, Matrix<TFloat> grad,
                                    TFloat learning_rate) {
  if (m.rows() != grad.rows() || m.cols() != grad.cols())
    throw exception();

  for (size_t i = 0; i != m.data.size(); i++)
    dynamic_pointer_cast<VariableExpression>(m.data[i])->set_value(
        m.data[i]->value() - grad.data[i] * learning_rate);
}

void load_values_in_value_expression_matrix(Matrix<Expression> &m,
                                            const Matrix<TFloat> &values) {
  if (m.rows() != values.rows() || m.cols() != values.cols())
    throw exception();

  for (size_t i = 0; i != m.data.size(); i++)
    dynamic_pointer_cast<VariableExpression>(m.data[i])->set_value(
        values.data[i]);
}

NeuralNetwork::NeuralNetwork(size_t inputs_count, size_t hidden_layer_size,
                             size_t outputs_count,
                             const ActivationFunction &activation_function,
                             const ErrorFunction &error_function)
    : W1(randomize_variable_matrix(hidden_layer_size, inputs_count)),
      W2(randomize_variable_matrix(outputs_count, hidden_layer_size)),
      B1(randomize_variable_matrix(hidden_layer_size, 1)),
      B2(randomize_variable_matrix(outputs_count, 1)),
      input(create_variable_matrix(inputs_count, 1)),
      expected_output(create_variable_matrix(outputs_count, 1)),
      inputs_count(inputs_count), hidden_layer_size(hidden_layer_size),
      outputs_count(outputs_count) {
  Matrix<Expression> Z1 = this->W1.dot(this->input) + this->B1;
  Matrix<Expression> A1 = activation_function.apply(Z1);
  Matrix<Expression> Z2 = this->W2.dot(A1) + this->B2;

  this->output = std::move(Z2);
  this->error = error_function.apply(this->output, this->expected_output);
}

Matrix<TFloat> NeuralNetwork::forward(const Matrix<TFloat> &input) {
  load_values_in_value_expression_matrix(this->input, input);

  Matrix<TFloat> result(this->outputs_count, 1);

  for (size_t i = 0; i != this->outputs_count; i++)
    result.data[i] = this->output.data[i]->value();

  return result;
}

void NeuralNetwork::backward(const Matrix<TFloat> &expected_output,
                             TFloat learning_rate) {
  load_values_in_value_expression_matrix(this->expected_output,
                                         expected_output);

  Matrix<TFloat> grad_W1 = grad(this->error, this->W1),
                 grad_W2 = grad(this->error, this->W2),
                 grad_B1 = grad(this->error, this->B1),
                 grad_B2 = grad(this->error, this->B2);

  update_value_expression_matrix(this->W1, grad_W1, learning_rate);
  update_value_expression_matrix(this->W2, grad_W2, learning_rate);
  update_value_expression_matrix(this->B1, grad_B1, learning_rate);
  update_value_expression_matrix(this->B2, grad_B2, learning_rate);
}
