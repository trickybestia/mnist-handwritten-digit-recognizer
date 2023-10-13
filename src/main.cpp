#include <format>
#include <iomanip>
#include <iostream>
#include <vector>

#include "layers/linear.hpp"

#include "layers/activation_functions/leaky_relu.hpp"
#include "layers/activation_functions/sigmoid.hpp"
#include "layers/activation_functions/tanh.hpp"

#include "error_functions/mean_squared_error.hpp"

#include "optimizers/adam.hpp"

#include "neural_network_builder.hpp"

const TFloat LEARNING_RATE = 0.03;
const TFloat BETA1 = 0.9;
const TFloat BETA2 = 0.999;

using namespace std;

vector<pair<Matrix, Matrix>>
make_dataset(const vector<pair<vector<TFloat>, TFloat>> &dataset) {
  vector<pair<Matrix, Matrix>> result;

  for (size_t i = 0; i != dataset.size(); i++) {
    Matrix inputs(dataset[i].first.size(), 1);

    for (size_t j = 0; j != inputs.size(); j++) {
      inputs(j) = dataset[i].first[j];
    }

    Matrix outputs(1, 1);

    outputs(0) = dataset[i].second;

    result.push_back({inputs, outputs});
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
  NeuralNetworkBuilder neural_network_builder =
      NeuralNetworkBuilder::create<MeanSquaredError>(2);

  neural_network_builder.add_layer<Linear>(2, 2);
  neural_network_builder.add_layer<Sigmoid>();
  neural_network_builder.add_layer<Linear>(2, 1);

  NeuralNetwork neural_network = neural_network_builder.build();
  Adam optimizer(neural_network, LEARNING_RATE, BETA1, BETA2);

  TFloat error = 0.0;

  for (size_t epoch = 0; epoch != 1000; epoch++) {
    for (size_t i = 0; i != DATASET.size(); i++) {
      neural_network.forward(DATASET[i].first);

      error += neural_network.expect(DATASET[i].second);

      neural_network.backward();
    }

    optimizer.step();

    if (epoch % 10 == 0) {
      error /= DATASET.size() * 10;

      cout << format("Epoch: {}; mean error: {}\n", epoch, error);

      error = 0.0;
    }
  }

  Matrix inputs(2, 1);

  while (cin >> inputs(0, 0) >> inputs(1, 0)) {
    cout << fixed << setprecision(numeric_limits<TFloat>::digits10 + 1)
         << neural_network.forward(inputs)(0, 0) << endl;
  }
}
