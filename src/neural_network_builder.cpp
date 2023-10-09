#include "neural_network_builder.hpp"

using namespace std;

NeuralNetworkBuilder::NeuralNetworkBuilder(
    size_t inputs_count, shared_ptr<ErrorFunction> error_function)
    : inputs_count(inputs_count), error_function(std::move(error_function)) {}

void NeuralNetworkBuilder::add_layer(size_t size) {
  this->layers.push_back({size, nullopt});
}

void NeuralNetworkBuilder::add_layer(
    size_t size, shared_ptr<ActivationFunction> activation_function) {
  this->layers.push_back(
      {size, std::make_optional(std::move(activation_function))});
}

NeuralNetwork NeuralNetworkBuilder::build() const {
  return NeuralNetwork(this->inputs_count, this->layers, this->error_function);
}
