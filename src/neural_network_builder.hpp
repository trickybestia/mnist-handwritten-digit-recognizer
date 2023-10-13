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

  template <typename T, typename... Args> void add_layer(Args... args) {
    this->_layers.push_back(std::make_shared<T>(args...));
  }

  NeuralNetwork build() const;
};
