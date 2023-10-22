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

using namespace std;

const bool TRAIN = false;
const bool TRAIN_EXISTING_MODEL = true;
const bool TRAIN_WITH_DROPOUT = true;

const TFloat RANDOM_PARAMETER_MIN = -0.3;
const TFloat RANDOM_PARAMETER_MAX = 0.3;

const TFloat LEARNING_RATE = 0.003;
const TFloat BETA1 = 0.9;
const TFloat BETA2 = 0.999;

const size_t SAMPLES_BETWEEN_LOG = 1000;

size_t max_item_index(const Matrix &matrix) {
  size_t result = 0;

  for (size_t i = 0; i != matrix.size(); i++) {
    if (matrix(i) > matrix(result)) {
      result = i;
    }
  }

  return result;
}

NeuralNetwork create_neural_network() {
  auto cross_entropy_softmax = make_shared<CrossEntropySoftmax>();

  NeuralNetworkBuilder neural_network_builder(cross_entropy_softmax);

  if (TRAIN && TRAIN_WITH_DROPOUT)
    neural_network_builder.add_layer(make_shared<Dropout>(0.2));

  neural_network_builder.add_layer(make_shared<Linear>(784, 512));
  neural_network_builder.add_layer(make_shared<Sigmoid>());

  neural_network_builder.add_layer(make_shared<Linear>(512, 128));
  neural_network_builder.add_layer(make_shared<Sigmoid>());

  neural_network_builder.add_layer(make_shared<Linear>(128, 64));
  neural_network_builder.add_layer(make_shared<Sigmoid>());

  neural_network_builder.add_layer(make_shared<Linear>(64, 10));
  neural_network_builder.add_layer(cross_entropy_softmax);

  return neural_network_builder.build();
}

TFloat compute_test_accuracy(NeuralNetwork &neural_network,
                             const mnist::Dataset &dataset) {
  size_t correctly_predicted = 0;

  for (const mnist::DatasetEntry &entry : dataset.test_entries()) {
    Matrix output = neural_network.forward(entry.image());
    size_t digit = max_item_index(output);

    if (digit == entry.label()) {
      correctly_predicted++;
    }
  }

  return static_cast<TFloat>(correctly_predicted) /
         dataset.test_entries().size();
}

void showcase(NeuralNetwork &neural_network) {
  Matrix input(28 * 28, 1);

  while (true) {
    load_matrix(input, "image.bin");

    Matrix output = neural_network.forward(input);

    for (size_t j = 0; j != output.size(); j++) {
      cout << format("{}: {}", j, output(j)) << endl;
    }

    cin.get();
  }
}

int main() {
  auto dataset = mnist::load_dataset("/home/trickybestia/Downloads/mnist/");

  NeuralNetwork neural_network = create_neural_network();

  if (!TRAIN || TRAIN_EXISTING_MODEL) {
    load_matrix(neural_network.parameters(), "model.bin");
  }

  if (!TRAIN) {
    cout << format("Test set accuracy: {}\n",
                   compute_test_accuracy(neural_network, dataset));

    showcase(neural_network);

    return 0;
  }

  if (!TRAIN_EXISTING_MODEL) {
    neural_network.randomize_parameters(RANDOM_PARAMETER_MIN,
                                        RANDOM_PARAMETER_MAX);
  }

  // Adam optimizer(neural_network, LEARNING_RATE, BETA1, BETA2);
  SGD optimizer(neural_network, LEARNING_RATE);
  // Momentum optimizer(neural_network, LEARNING_RATE);

  TFloat mean_error = 0.0;
  size_t correctly_predicted = 0;

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
      save_matrix(neural_network.parameters(), "model.bin");

      cout << "Parameters saved\n";
    }

    const mnist::DatasetEntry &entry =
        dataset.train_entries()[j % dataset.train_entries().size()];

    Matrix output = neural_network.forward(entry.image());
    size_t digit = max_item_index(output);

    if (digit == entry.label()) {
      correctly_predicted++;
    }

    Matrix expected_output(10, 1);
    expected_output.zeroize();
    expected_output(entry.label()) = 1.0;

    mean_error += neural_network.expect(std::move(expected_output));

    neural_network.backward();

    optimizer.step();

    neural_network.gradient().zeroize();
  }
}
