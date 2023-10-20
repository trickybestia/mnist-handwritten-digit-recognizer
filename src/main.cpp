#include <format>
#include <iostream>

#include "layers/linear.hpp"

#include "layers/activation_functions/leaky_relu.hpp"
#include "layers/activation_functions/sigmoid.hpp"
#include "layers/activation_functions/tanh.hpp"

#include "layers/dropout.hpp"

#include "error_functions/cross_entropy_softmax.hpp"
#include "error_functions/mean.hpp"
#include "error_functions/mean_squared.hpp"

#include "optimizers/adam.hpp"
#include "optimizers/momentum.hpp"
#include "optimizers/sgd.hpp"

#include "neural_network_builder.hpp"

#include "mnist/load_dataset.hpp"

#include "utils.hpp"

const TFloat LEARNING_RATE = 0.01;
const TFloat BETA1 = 0.9;
const TFloat BETA2 = 0.999;

const size_t SAMPLES_BETWEEN_LOG = 1000;

using namespace std;

void showcase(NeuralNetwork &neural_network, const mnist::Dataset &dataset) {
  for (size_t i = 0; i != dataset.train_entries().size(); i++) {
    const mnist::DatasetEntry &entry = dataset.train_entries()[i];

    cout << static_cast<int>(entry.label()) << endl;

    Matrix output = neural_network.forward(
        Eigen::Map<const Vector>(entry.image().data(), entry.image().size()));

    for (ssize_t j = 0; j != output.size(); j++) {
      cout << format("{}: {}", j, output(j)) << endl;
    }

    cin.get();
  }
}

int main() {
  auto dataset = mnist::load_dataset("/home/trickybestia/Downloads/mnist/");

  auto cross_entropy_softmax = make_shared<CrossEntropySoftmax>();

  NeuralNetworkBuilder neural_network_builder(cross_entropy_softmax);

  neural_network_builder.add_layer(make_shared<Linear>(784, 512));
  neural_network_builder.add_layer(make_shared<Sigmoid>());

  // neural_network_builder.add_layer(make_shared<Dropout>(0.25));

  neural_network_builder.add_layer(make_shared<Linear>(512, 128));
  neural_network_builder.add_layer(make_shared<Sigmoid>());

  neural_network_builder.add_layer(make_shared<Linear>(128, 64));
  neural_network_builder.add_layer(make_shared<Sigmoid>());

  neural_network_builder.add_layer(make_shared<Linear>(64, 10));
  neural_network_builder.add_layer(cross_entropy_softmax);

  NeuralNetwork neural_network = neural_network_builder.build();
  // Adam optimizer(neural_network, LEARNING_RATE, BETA1, BETA2);
  SGD optimizer(neural_network, LEARNING_RATE);
  // Momentum optimizer(neural_network, LEARNING_RATE);

  // load_parameters(neural_network.parameters(), "model.bin");

  // showcase(neural_network, dataset);

  // return 0;

  size_t correctly_predicted = 0;
  TFloat mean_error = 0.0;

  for (size_t j = 0;; j++) {
    if (j != 0 && j % SAMPLES_BETWEEN_LOG == 0) {
      cout << format("Samples: {:6}; epoch: {:6}; accuracy: "
                     "{:.4f}; mean error: {:.7f}\n",
                     j, j / dataset.train_entries().size(),
                     static_cast<TFloat>(correctly_predicted) /
                         SAMPLES_BETWEEN_LOG,
                     mean_error / SAMPLES_BETWEEN_LOG);

      correctly_predicted = 0;
      mean_error = 0.0;
    }

    if (j != 0 && j % 30000 == 0) {
      save_parameters(neural_network.parameters(), "model.bin");

      cout << "Parameters saved\n";
    }

    const mnist::DatasetEntry &entry =
        dataset.train_entries()[j % dataset.train_entries().size()];

    Vector output = neural_network.forward(
        Eigen::Map<const Vector>(entry.image().data(), entry.image().size()));
    size_t output_max_index = 0;

    for (ssize_t i = 1; i != output.size(); i++) {
      if (output(i) > output(output_max_index)) {
        output_max_index = i;
      }
    }

    if (output_max_index == entry.label()) {
      correctly_predicted++;
    }

    Vector expected_output = Vector::Zero(10);
    expected_output(entry.label()) = 1.0;

    mean_error += neural_network.expect(expected_output);

    neural_network.backward();

    optimizer.step();

    neural_network.gradient().setZero();
  }
}
