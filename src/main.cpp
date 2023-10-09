#include <cmath>
#include <format>
#include <iomanip>
#include <iostream>
#include <vector>

#include "activation_functions/leaky_relu.hpp"
#include "activation_functions/sigmoid.hpp"
#include "activation_functions/tanh.hpp"
#include "error_functions/mean_squared_error.hpp"
#include "neural_network.hpp"

const TFloat LEARNING_RATE = 0.03;

using namespace std;

vector<pair<Matrix<TFloat>, Matrix<TFloat>>>
make_dataset(const vector<pair<vector<TFloat>, TFloat>> &dataset) {
  vector<pair<Matrix<TFloat>, Matrix<TFloat>>> result;

  for (size_t i = 0; i != dataset.size(); i++) {
    Matrix<TFloat> inputs(dataset[i].first.size(), 1);

    inputs.data = dataset[i].first;

    Matrix<TFloat> outputs(1, 1);

    outputs(0, 0) = dataset[i].second;

    result.push_back({std::move(inputs), std::move(outputs)});
  }

  return result;
}

int main() {
  NeuralNetwork neural_network(2, 3000, 1, make_unique<LeakyReLU>(0.01),
                               make_unique<MeanSquaredError>());

  vector<pair<Matrix<TFloat>, Matrix<TFloat>>> dataset =
      make_dataset({{{1, 1}, 0}, {{1, 0}, 1}, {{0, 1}, 1}, {{0, 0}, 0}});

  for (size_t epoch = 0; epoch != 1000; epoch++) {
    TFloat error = 0.0;

    for (size_t i = 0; i != dataset.size(); i++) {
      neural_network.forward(dataset[i].first);

      error += neural_network.expect(dataset[i].second);

      neural_network.backward(LEARNING_RATE);
    }

    error /= dataset.size();

    if (epoch % 10 == 0) {
      cout << format("Epoch: {}; Mean error: {}", epoch, error) << endl;
    }
  }

  Matrix<TFloat> inputs(2, 1);

  while (cin >> inputs(0, 0) >> inputs(1, 0)) {
    cout << fixed << setprecision(numeric_limits<TFloat>::digits10 + 1)
         << neural_network.forward(inputs)(0, 0) << endl;
  }
}
