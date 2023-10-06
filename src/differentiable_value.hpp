#pragma once

#include <cstdint>
#include <unordered_map>
#include <unordered_set>

typedef double TFloat;
typedef uint32_t VariableId;

class DifferentiableValue {
private:
  std::unordered_map<VariableId, TFloat> partial_derivatives;
  TFloat _value;

  std::unordered_set<VariableId>
  union_partial_derivatives_keys(const DifferentiableValue &other);

public:
  DifferentiableValue();
  DifferentiableValue(TFloat value);
  DifferentiableValue(TFloat value, VariableId variable);

  TFloat derivative(VariableId wrt) const;

  inline TFloat value() const { return this->_value; };

  friend DifferentiableValue pow(const DifferentiableValue &x, TFloat power);
  friend DifferentiableValue pow(TFloat base, const DifferentiableValue &power);
  friend DifferentiableValue abs(const DifferentiableValue &x);

  DifferentiableValue operator-() const;

  void operator+=(const DifferentiableValue &other);
  void operator-=(const DifferentiableValue &other);
  void operator*=(const DifferentiableValue &other);
  void operator/=(const DifferentiableValue &other);

  DifferentiableValue operator+(const DifferentiableValue &other) const;
  DifferentiableValue operator-(const DifferentiableValue &other) const;
  DifferentiableValue operator*(const DifferentiableValue &other) const;
  DifferentiableValue operator/(const DifferentiableValue &other) const;
};

DifferentiableValue operator""_diff(long double);
