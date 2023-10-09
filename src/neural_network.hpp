#pragma once

#include <memory>
#include <optional>

#include "activation_function.hpp"
#include "error_function.hpp"

class NeuralNetwork {
public:
  struct LinearLayer {
    Matrix weights, bias, weights_grad, bias_grad;
    std::optional<std::shared_ptr<ActivationFunction>> activation_function;

    std::optional<Matrix> input;

    LinearLayer(
        size_t inputs, size_t outputs,
        std::optional<std::shared_ptr<ActivationFunction>> activation_function);

    Matrix forward();
    void backward(const Matrix &layer_error);

    Matrix previous_layer_error(const Matrix &layer_error);

    void flush_gradients(TFloat learning_rate);
  };

private:
  std::vector<LinearLayer> layers;

  Matrix output, expected_output;

  std::shared_ptr<ActivationFunction> activation_function;
  std::shared_ptr<ErrorFunction> error_function;

  void randomize_parameters(TFloat mean, TFloat stddev);

public:
  const size_t inputs_count;

  NeuralNetwork(
      size_t inputs_count,
      std::vector<
          std::pair<size_t, std::optional<std::shared_ptr<ActivationFunction>>>>
          layers,
      std::shared_ptr<ErrorFunction> error_function);

  Matrix forward(const Matrix &input);

  TFloat expect(const Matrix &expected_output);
  void backward(TFloat learning_rate);
};
