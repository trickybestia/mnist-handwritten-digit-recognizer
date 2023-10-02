#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "neural_network.hpp"

using namespace std;

class Sigmoid : public ActivationFunction {
public:
  virtual TFloat apply(TFloat parameter) const override {
    return 1.0 / (1.0 + exp(-parameter));
  }

  virtual TFloat derivative(TFloat parameter) const override {
    TFloat sigmoid = this->apply(parameter);

    return sigmoid * (1.0 - sigmoid);
  }
};

class MeanSquaredError : public ErrorFunction {
  virtual TFloat apply(const Matrix &got,
                       const Matrix &expected) const override {
    if (got.rows() != expected.rows() || got.cols() != expected.cols())
      throw exception();

    TFloat sum = 0.0;

    for (size_t i = 0; i != got.data.size(); i++) {
      sum += pow(got.data[i] - expected.data[i], 2);
    }

    return sum * 2.0 / got.data.size();
  }

  virtual Matrix derivative(const Matrix &got,
                            const Matrix &expected) const override {
    return got - expected;
  }
};

vector<pair<Matrix, Matrix>>
make_dataset(const vector<pair<vector<TFloat>, TFloat>> &dataset) {
  vector<pair<Matrix, Matrix>> result;

  for (size_t i = 0; i != dataset.size(); i++) {
    Matrix inputs(1, dataset[i].first.size());

    inputs.data = dataset[i].first;

    Matrix outputs(1, 1);

    outputs(0, 0) = dataset[i].second;

    result.push_back({std::move(inputs), std::move(outputs)});
  }

  return result;
}

int main() {
  NeuralNetwork neural_network(2, 3, 1, make_unique<Sigmoid>(),
                               make_unique<MeanSquaredError>());

  vector<pair<Matrix, Matrix>> dataset =
      make_dataset({{{1, 1}, 0}, {{1, 0}, 1}, {{0, 1}, 1}, {{0, 0}, 0}});

  for (size_t epoch = 0; epoch != 1000; epoch++) {
    for (size_t i = 0; i != dataset.size(); i++) {
      neural_network.forward(dataset[i].first);
      neural_network.backward(dataset[i].second, 0.1);
    }
  }

  while (true) {
    Matrix matrix(1, 2);

    cin >> matrix(0, 0) >> matrix(0, 1);

    cout << neural_network.forward(matrix)(0, 0) << endl;
  }
}
