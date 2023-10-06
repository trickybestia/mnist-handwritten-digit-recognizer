#include <cmath>

#include "differentiable_value.hpp"

using namespace std;

DifferentiableValue operator""_diff(long double value) {
  return DifferentiableValue(value);
}

DifferentiableValue pow(const DifferentiableValue &x, TFloat power) {
  DifferentiableValue result = x;

  TFloat derivative = power * pow(result._value, power - 1.0);

  result._value = pow(result._value, power);

  for (size_t i = 0; i != result.derivatives.size(); i++)
    result.derivatives[i] *= derivative;

  return result;
}

DifferentiableValue pow(TFloat base, const DifferentiableValue &power) {
  DifferentiableValue result = power;

  result._value = pow(base, result._value);

  TFloat derivative = result._value * log(base);

  for (size_t i = 0; i != result.derivatives.size(); i++)
    result.derivatives[i] *= derivative;

  return result;
}

DifferentiableValue abs(const DifferentiableValue &x) {
  if (x._value < 0)
    return -x;

  return x;
}

DifferentiableValue::DifferentiableValue() {}

DifferentiableValue::DifferentiableValue(TFloat value) : _value(value) {}

DifferentiableValue::DifferentiableValue(TFloat value, VariableId variable)
    : derivatives(variable + 1), _value(value) {
  this->derivatives[variable] = 1.0;
}

void DifferentiableValue::align_size(const DifferentiableValue &other) {
  if (other.derivatives.size() > this->derivatives.size())
    this->derivatives.resize(other.derivatives.size());
}

TFloat DifferentiableValue::derivative(VariableId wrt) const {
  if (wrt >= this->derivatives.size())
    return 0.0;

  return this->derivatives[wrt];
}

DifferentiableValue DifferentiableValue::operator-() const {
  DifferentiableValue result = *this;

  result._value = -result._value;

  for (size_t i = 0; i != result.derivatives.size(); i++)
    result.derivatives[i] = -result.derivatives[i];

  return result;
}

void DifferentiableValue::operator+=(const DifferentiableValue &other) {
  this->align_size(other);

  this->_value += other._value;

  for (size_t i = 0; i != other.derivatives.size(); i++)
    this->derivatives[i] += other.derivatives[i];
}

void DifferentiableValue::operator-=(const DifferentiableValue &other) {
  this->align_size(other);

  this->_value -= other._value;

  for (size_t i = 0; i != other.derivatives.size(); i++)
    this->derivatives[i] -= other.derivatives[i];
}

void DifferentiableValue::operator*=(const DifferentiableValue &other) {
  this->align_size(other);

  for (size_t i = 0; i != this->derivatives.size(); i++)
    this->derivatives[i] = this->_value * other.derivative(i) +
                           this->derivatives[i] * other._value;

  this->_value *= other._value;
}

void DifferentiableValue::operator/=(const DifferentiableValue &other) {
  this->align_size(other);

  TFloat squared_other_value = pow(other._value, 2);

  for (size_t i = 0; i != this->derivatives.size(); i++)
    this->derivatives[i] = (this->derivatives[i] * other._value -
                            other.derivative(i) * this->_value) /
                           squared_other_value;

  this->_value /= other._value;
}

DifferentiableValue
DifferentiableValue::operator+(const DifferentiableValue &other) const {
  auto result = *this;

  result += other;

  return result;
}

DifferentiableValue
DifferentiableValue::operator-(const DifferentiableValue &other) const {
  auto result = *this;

  result -= other;

  return result;
}

DifferentiableValue
DifferentiableValue::operator*(const DifferentiableValue &other) const {
  auto result = *this;

  result *= other;

  return result;
}

DifferentiableValue
DifferentiableValue::operator/(const DifferentiableValue &other) const {
  auto result = *this;

  result /= other;

  return result;
}
