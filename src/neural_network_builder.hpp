#pragma once

#include "neural_network.hpp"

class NeuralNetworkBuilder {
private:
  size_t _inputs_count;
  std::vector<std::shared_ptr<Layer>> _layers;
  std::shared_ptr<ErrorFunction> _error_function;

  NeuralNetworkBuilder(size_t inputs_count,
                       std::shared_ptr<ErrorFunction> error_function);

public:
  template <typename T, typename... Args>
  static NeuralNetworkBuilder create(size_t inputs_count, Args... args) {
    return NeuralNetworkBuilder(inputs_count, std::make_shared<T>(args...));
  }

  template <typename T, typename... Args> void add_layer(Args... args) {
    this->_layers.push_back(std::make_shared<T>(args...));
  }

  NeuralNetwork build() const;
};
