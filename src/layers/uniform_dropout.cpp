#include "uniform_dropout.hpp"

using namespace std;

UniformDropout::UniformDropout(float probability, TFloat min, TFloat max)
    : _probability_distribution(probability), _value_distribution(min, max) {
  random_device rd;

  this->_rng.seed(rd());
}

size_t UniformDropout::parameters_count() const { return 0; }

void UniformDropout::set_parameters(TFloat *) {}

void UniformDropout::set_gradient(TFloat *) {}

void UniformDropout::backward(const Matrix &) {}

Matrix UniformDropout::forward(Matrix input) {
  Matrix result(input.rows(), input.cols());

  this->_factors = Matrix(input.rows(), input.cols());

  for (size_t i = 0; i != input.size(); i++) {
    if (this->_probability_distribution(this->_rng)) {
      result(i) = this->_value_distribution(this->_rng);

      this->_factors(i) = 0.0;
    } else {
      result(i) = input(i);

      this->_factors(i) = 1.0;
    }
  }

  return result;
}

Matrix UniformDropout::previous_layer_error(const Matrix &layer_error) {
  return layer_error * this->_factors;
}
