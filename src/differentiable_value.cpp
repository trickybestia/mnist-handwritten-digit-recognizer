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

  for (auto i = result.partial_derivatives.begin();
       i != result.partial_derivatives.end(); i++) {
    i->second *= derivative;
  }

  return result;
}

DifferentiableValue pow(TFloat base, const DifferentiableValue &power) {
  DifferentiableValue result = power;

  result._value = pow(base, result._value);

  TFloat derivative = result._value * log(base);

  for (auto i = result.partial_derivatives.begin();
       i != result.partial_derivatives.end(); i++) {
    i->second *= derivative;
  }

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
    : partial_derivatives({{variable, 1.0}}), _value(value) {}

std::unordered_set<VariableId>
DifferentiableValue::union_partial_derivatives_keys(
    const DifferentiableValue &other) {
  unordered_set<VariableId> result;

  for (auto [varaible_id, _] : this->partial_derivatives) {
    result.insert(varaible_id);
  }

  for (auto [varaible_id, _] : other.partial_derivatives) {
    result.insert(varaible_id);
  }

  return result;
}

TFloat DifferentiableValue::derivative(VariableId wrt) const {
  auto i = this->partial_derivatives.find(wrt);

  if (i != this->partial_derivatives.end()) {
    return i->second;
  }

  return 0.0;
}

DifferentiableValue DifferentiableValue::operator-() const {
  DifferentiableValue result = *this;

  result._value = -result._value;

  for (auto i = result.partial_derivatives.begin();
       i != result.partial_derivatives.end(); i++) {
    i->second = -i->second;
  }

  return result;
}

void DifferentiableValue::operator+=(const DifferentiableValue &other) {
  this->_value += other._value;

  for (auto i = other.partial_derivatives.begin();
       i != other.partial_derivatives.end(); i++) {
    this->partial_derivatives[i->first] =
        this->derivative(i->first) + i->second;
  }
}

void DifferentiableValue::operator-=(const DifferentiableValue &other) {
  this->_value -= other._value;

  for (auto i = other.partial_derivatives.begin();
       i != other.partial_derivatives.end(); i++) {
    this->partial_derivatives[i->first] =
        this->derivative(i->first) - i->second;
  }
}

void DifferentiableValue::operator*=(const DifferentiableValue &other) {
  for (VariableId variable : this->union_partial_derivatives_keys(other)) {
    this->partial_derivatives[variable] =
        this->_value * other.derivative(variable) +
        this->derivative(variable) * other._value;
  }

  this->_value *= other._value;
}

void DifferentiableValue::operator/=(const DifferentiableValue &other) {
  TFloat squared_other_value = pow(other._value, 2);

  for (VariableId variable : this->union_partial_derivatives_keys(other)) {
    this->partial_derivatives[variable] =
        (this->derivative(variable) * other._value -
         other.derivative(variable) * this->_value) /
        squared_other_value;
  }

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
