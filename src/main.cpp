#include <format>
#include <iomanip>
#include <iostream>
#include <vector>

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

const TFloat LEARNING_RATE = 0.003;
const TFloat BETA1 = 0.9;
const TFloat BETA2 = 0.999;

const size_t DATASET_SIZE = 60000;

using namespace std;

void showcase(NeuralNetwork &neural_network, const mnist::Dataset &dataset) {
  for (size_t i = 0; i != dataset.train_entries().size(); i++) {
    const mnist::DatasetEntry &entry = dataset.train_entries()[i];

    Matrix output = neural_network.forward(entry.image());

    cout << static_cast<int>(entry.label()) << endl;

    for (size_t j = 0; j != output.size(); j++) {
      cout << format("{}: {}", j, output(j)) << endl;
    }

    cin.get();
  }
}

int main() {
  auto dataset = mnist::load_dataset("/home/trickybestia/Downloads/mnist/");

  auto cross_entropy_softmax = make_shared<CrossEntropySoftmax>();

  NeuralNetworkBuilder neural_network_builder(784, cross_entropy_softmax);

  neural_network_builder.add_layer(make_shared<Linear>(784, 300));
  neural_network_builder.add_layer(make_shared<LeakyReLU>(0.01));

  // neural_network_builder.add_layer(make_shared<Dropout>(0.25));

  neural_network_builder.add_layer(make_shared<Linear>(300, 10));
  neural_network_builder.add_layer(cross_entropy_softmax);

  NeuralNetwork neural_network = neural_network_builder.build();
  // Adam optimizer(neural_network, LEARNING_RATE, BETA1, BETA2);
  SGD optimizer(neural_network, LEARNING_RATE);
  // Momentum optimizer(neural_network, LEARNING_RATE);

  load_parameters(neural_network.parameters(), "model.bin");

  showcase(neural_network, dataset);

  // return 0;

  size_t j = 0;
  TFloat mean_error = 0.0;

  for (size_t epoch = 0;; epoch++) {
    for (size_t i = 0; i != DATASET_SIZE; i++, j++) {
      const mnist::DatasetEntry &entry = dataset.train_entries()[i];

      neural_network.forward(entry.image());

      Matrix expected_output(10, 1);
      expected_output.zeroize();
      expected_output(entry.label()) = 1.0;

      neural_network.expect(std::move(expected_output));

      mean_error += neural_network.value();

      neural_network.backward();

      if (j % 128 == 0) {
        neural_network.gradient() /= 128;

        optimizer.step();

        neural_network.gradient().zeroize();
      }

      if (j % 10000 == 9999) {
        cout << format("Epoch: {:6}; i: {:6}; mean error: {}\n", epoch, i,
                       mean_error / 10000);

        save_parameters(neural_network.parameters(), "model.bin");

        mean_error = 0.0;
      }
    }
  }

  showcase(neural_network, dataset);

  /*Matrix inputs(2, 1);

  while (cin >> inputs(0, 0) >> inputs(1, 0)) {
    cout << fixed << setprecision(numeric_limits<TFloat>::digits10 + 1)
         << neural_network.forward(inputs)(0, 0) << endl;
  }*/
}
