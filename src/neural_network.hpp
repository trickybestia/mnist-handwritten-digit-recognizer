#pragma once

#include <memory>

#include "error_function.hpp"
#include "function.hpp"
#include "layer.hpp"

class NeuralNetwork : public Function {
public:
private:
  std::vector<std::shared_ptr<Layer>> _layers;

  Matrix _output, _expected_output, _parameters, _gradient;

  std::shared_ptr<ErrorFunction> _error_function;

  void randomize_parameters(TFloat mean, TFloat stddev);

public:
  const size_t inputs_count;

  NeuralNetwork(size_t inputs_count, std::vector<std::shared_ptr<Layer>> layers,
                std::shared_ptr<ErrorFunction> error_function);

  Matrix forward(Matrix input);

  TFloat expect(Matrix expected_output);

  void backward();

  virtual Matrix &parameters() override;
  virtual Matrix &gradient() override;
};
