#pragma once

#include <memory>

#include "error_function.hpp"
#include "function.hpp"
#include "layer.hpp"

class NeuralNetwork : public Function {
private:
  std::vector<std::shared_ptr<Layer>> _layers;

  Vector _output, _expected_output, _parameters, _gradient;

  std::shared_ptr<ErrorFunction> _error_function;

  void randomize_parameters(TFloat mean, TFloat stddev);

public:
  NeuralNetwork(std::vector<std::shared_ptr<Layer>> layers,
                std::shared_ptr<ErrorFunction> error_function);

  Vector forward(const Vector &input);

  TFloat expect(const Vector &expected_output);

  void backward();

  virtual Vector &parameters() override;
  virtual Vector &gradient() override;
};
