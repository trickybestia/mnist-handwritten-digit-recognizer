#include "neural_network_builder.hpp"
#include "layers/linear.hpp"

using namespace std;

NeuralNetworkBuilder::NeuralNetworkBuilder(
    size_t inputs_count, shared_ptr<ErrorFunction> error_function)
    : _inputs_count(inputs_count), _error_function(std::move(error_function)) {}

void NeuralNetworkBuilder::add_layer(shared_ptr<Layer> layer) {
  this->_layers.push_back(std::move(layer));
}

NeuralNetwork NeuralNetworkBuilder::build() const {
  return NeuralNetwork(this->_inputs_count, this->_layers,
                       this->_error_function);
}
