#pragma once

#include "neural_network.hpp"

class NeuralNetworkBuilder {
private:
  size_t inputs_count;
  std::vector<
      std::pair<size_t, std::optional<std::shared_ptr<ActivationFunction>>>>
      layers;
  std::shared_ptr<ErrorFunction> error_function;

public:
  NeuralNetworkBuilder(size_t inputs_count,
                       std::shared_ptr<ErrorFunction> error_function);

  void add_layer(size_t size);
  void add_layer(size_t size,
                 std::shared_ptr<ActivationFunction> activation_function);

  NeuralNetwork build() const;
};
