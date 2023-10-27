#include <format>

#include <tclap/CmdLine.h>

#include "args.hpp"

using namespace TCLAP;
using namespace std;

Args parse_args(int argc, char **argv) {
  CmdLine cmdline("DIY neural network for digit recognition");

  ValueArg<string> model_arg("m", "model", "Path to model file", true, "",
                             "path", cmdline);
  ValueArg<string> dataset_arg("d", "dataset",
                               "Path to MNIST Handwritten dataset root", false,
                               "", "path", cmdline);
  ValuesConstraint<string> optimizer_arg_constraint(
      {"SGD", "Adam", "Momentum"});
  ValueArg<string> optimizer_arg("o", "optimizer",
                                 "Name of optimizer used in learning", false,
                                 "", &optimizer_arg_constraint, cmdline);
  SwitchArg train_arg("t", "train", "Train model");
  SwitchArg showcase_arg("s", "showcase", "Run in showcase mode");
  SwitchArg train_existing_model_arg("e", "train-existing-model",
                                     "Train existing model", cmdline);
  SwitchArg train_with_dropout_arg("p", "train-with-dropout",
                                   "Train model with dropout", cmdline);
  SwitchArg compute_test_accuracy_arg("a", "compute-test-accuracy",
                                      "Compute model accuracy on test dataset");
  ValueArg<TFloat> learning_rate_arg("l", "learning-rate",
                                     "Learning rate [DEFAULT: 0.01]", false,
                                     0.01, "float", cmdline);
  ValueArg<TFloat> random_parameter_min_arg(
      "", "random-parameter-min",
      "Min value of randomized model parameter [DEFAULT: -0.3]", false, -0.3,
      "float", cmdline);
  ValueArg<TFloat> random_parameter_max_arg(
      "", "random-parameter-max",
      "Max value of randomized model parameter [DEFAULT: 0.3]", false, 0.3,
      "float", cmdline);
  ValueArg<TFloat> beta1_arg("", "beta1",
                             "Beta hyperparameter for Momentum optimizer, or "
                             "beta1 for Adam [DEFAULT: 0.9]",
                             false, 0.9, "float", cmdline);
  ValueArg<TFloat> beta2_arg(
      "", "beta2", "Beta2 hyperparameter for Adam optimizer [DEFAULT: 0.999]",
      false, 0.999, "float", cmdline);

  cmdline.xorAdd({&train_arg, &showcase_arg, &compute_test_accuracy_arg});

  cmdline.parse(argc, argv);

  return Args{
      .model = model_arg.getValue(),
      .dataset =
          dataset_arg.isSet() ? make_optional(dataset_arg.getValue()) : nullopt,
      .optimizer = optimizer_arg.isSet()
                       ? make_optional(optimizer_arg.getValue())
                       : nullopt,
      .train = train_arg.getValue(),
      .train_existing_model = train_existing_model_arg.getValue(),
      .train_with_dropout = train_with_dropout_arg.getValue(),
      .compute_test_accuracy = compute_test_accuracy_arg.getValue(),
      .showcase = showcase_arg.getValue(),
      .learning_rate = learning_rate_arg.getValue(),
      .random_parameter_min = random_parameter_min_arg.getValue(),
      .random_parameter_max = random_parameter_max_arg.getValue(),
      .beta1 = beta1_arg.getValue(),
      .beta2 = beta2_arg.getValue(),
  };
}
