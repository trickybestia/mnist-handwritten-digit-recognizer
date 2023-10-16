#pragma once

#include "neural_network.hpp"

class NeuralNetworkBuilder {
private:
  size_t _inputs_count;
  std::vector<std::shared_ptr<Layer>> _layers;
  std::shared_ptr<ErrorFunction> _error_function;

public:
  NeuralNetworkBuilder(size_t inputs_count,
                       std::shared_ptr<ErrorFunction> error_function);

  void add_layer(std::shared_ptr<Layer> layer);

  NeuralNetwork build() const;
};
