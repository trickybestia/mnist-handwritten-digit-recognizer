#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "activation_functions/sigmoid.hpp"
#include "error_functions/mean_squared_error.hpp"
#include "neural_network.hpp"

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
  NeuralNetwork neural_network(2, 3, 1, Sigmoid(), MeanSquaredError());

  vector<pair<Matrix<TFloat>, Matrix<TFloat>>> dataset =
      make_dataset({{{1, 1}, 0}, {{1, 0}, 1}, {{0, 1}, 1}, {{0, 0}, 0}});

  for (size_t epoch = 0; epoch != 1000; epoch++) {
    cout << epoch << endl;

    for (size_t i = 0; i != dataset.size(); i++) {
      neural_network.forward(dataset[i].first);
      neural_network.backward(dataset[i].second, 0.1);
    }
  }

  Matrix<TFloat> inputs(2, 1);

  while (cin >> inputs(0, 0) >> inputs(1, 0)) {
    cout << fixed << setprecision(numeric_limits<TFloat>::digits10 + 1)
         << neural_network.forward(inputs)(0, 0) << endl;
  }
}
