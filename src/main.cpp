#include <cmath>
#include <format>
#include <iomanip>
#include <iostream>
#include <vector>

#include "activation_functions/leaky_relu.hpp"
#include "activation_functions/sigmoid.hpp"
#include "activation_functions/tanh.hpp"
#include "error_functions/mean_squared_error.hpp"
#include "neural_network_builder.hpp"

const TFloat LEARNING_RATE = 0.03;

using namespace std;

vector<pair<Matrix, Matrix>>
make_dataset(const vector<pair<vector<TFloat>, TFloat>> &dataset) {
  vector<pair<Matrix, Matrix>> result;

  for (size_t i = 0; i != dataset.size(); i++) {
    Matrix inputs(dataset[i].first.size(), 1);

    inputs.data = dataset[i].first;

    Matrix outputs(1, 1);

    outputs(0, 0) = dataset[i].second;

    result.push_back({std::move(inputs), std::move(outputs)});
  }

  return result;
}

const vector<pair<Matrix, Matrix>> DATASET = make_dataset({
    {{0, 0}, 0},
    {{0, 1}, 1},
    {{1, 0}, 1},
    {{1, 1}, 0},
});

int main() {
  NeuralNetworkBuilder neural_network_builder(2,
                                              make_shared<MeanSquaredError>());

  neural_network_builder.add_layer(3, make_shared<LeakyReLU>(0.05));
  neural_network_builder.add_layer(1);

  NeuralNetwork neural_network = neural_network_builder.build();

  for (size_t epoch = 0; epoch != 1000; epoch++) {
    TFloat error = 0.0;

    for (size_t i = 0; i != DATASET.size(); i++) {
      neural_network.forward(DATASET[i].first);

      error += neural_network.expect(DATASET[i].second);

      neural_network.backward(LEARNING_RATE);
    }

    if (epoch % 10 == 0) {
      cout << format("Epoch: {}; mean error: {}", epoch, error / DATASET.size())
           << endl;
    }
  }

  Matrix inputs(2, 1);

  while (cin >> inputs(0, 0) >> inputs(1, 0)) {
    cout << fixed << setprecision(numeric_limits<TFloat>::digits10 + 1)
         << neural_network.forward(inputs)(0, 0) << endl;
  }
}
