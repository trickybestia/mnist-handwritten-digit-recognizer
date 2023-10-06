#pragma once

#include <memory>

#include "activation_function.hpp"
#include "error_function.hpp"

class NeuralNetwork {
private:
  std::pair<Matrix<DifferentiableValue>, VariableId> W1, W2, B1, B2;
  DifferentiableValue error;
  Matrix<DifferentiableValue> input, output;

  std::unique_ptr<ActivationFunction> activation_function;
  std::unique_ptr<ErrorFunction> error_function;

public:
  const size_t inputs_count, hidden_layer_size, outputs_count;

  NeuralNetwork(size_t inputs_count, size_t hidden_layer_size,
                size_t outputs_count,
                std::unique_ptr<ActivationFunction> activation_function,
                std::unique_ptr<ErrorFunction> error_function);

  Matrix<TFloat> forward(const Matrix<TFloat> &input);

  TFloat expect(const Matrix<TFloat> &expected_output);
  void backward(TFloat learning_rate);
};
