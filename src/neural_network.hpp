#pragma once

#include <memory>

#include "activation_function.hpp"
#include "error_function.hpp"
#include "float.hpp"
#include "matrix.hpp"

class NeuralNetwork {
private:
  Matrix A0, Z1, A1, Z2;

  void randomize();

public:
  Matrix W1, W2, B1, B2;

  const std::unique_ptr<ActivationFunction> activation_function;
  const std::unique_ptr<ErrorFunction> error_function;

  const size_t inputs_count, hidden_layer_size, outputs_count;

  NeuralNetwork(size_t inputs_count, size_t hidden_layer_size,
                size_t outputs_count,
                std::unique_ptr<ActivationFunction> activation_function,
                std::unique_ptr<ErrorFunction> error_function);

  const Matrix &forward(const Matrix &input);

  void backward(const Matrix &expected_output, TFloat learning_rate);
};
