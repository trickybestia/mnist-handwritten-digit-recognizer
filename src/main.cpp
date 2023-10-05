#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "neural_network.hpp"

using namespace std;

class Sigmoid : public ActivationFunction {
public:
  virtual Expression apply(Expression x) const override {
    return 1.0_expr / (1.0_expr + exp(M_E, -x));
  }
};

class MeanSquaredError : public ErrorFunction {
  virtual Expression apply(const Matrix<Expression> &got,
                           const Matrix<Expression> &expected) const override {
    if (got.rows() != expected.rows() || got.cols() != expected.cols())
      throw exception();

    Expression sum = pow(got.data[0] - expected.data[0], 2);

    for (size_t i = 1; i != got.data.size(); i++) {
      sum = sum + pow(got.data[i] - expected.data[i], 2);
    }

    return sum * 2.0_expr / make_shared<ValueExpression>(got.data.size());
  }
};

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

  while (true) {
    Matrix<TFloat> inputs(2, 1);

    cin >> inputs(0, 0) >> inputs(1, 0);

    cout << neural_network.forward(inputs)(0, 0) << endl;
  }
}
