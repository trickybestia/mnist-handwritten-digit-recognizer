#pragma once

#include <memory>

#include "activation_function.hpp"
#include "error_function.hpp"

class NeuralNetwork {
private:
  Matrix<Expression> W1, W2, B1, B2, input, output, expected_output;
  Expression error;

public:
  const size_t inputs_count, hidden_layer_size, outputs_count;

  NeuralNetwork(size_t inputs_count, size_t hidden_layer_size,
                size_t outputs_count,
                const ActivationFunction &activation_function,
                const ErrorFunction &error_function);

  Matrix<TFloat> forward(const Matrix<TFloat> &input);

  void backward(const Matrix<TFloat> &expected_output, TFloat learning_rate);
};
