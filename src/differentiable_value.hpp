#pragma once

#include <cstdint>
#include <vector>

typedef float TFloat;
typedef uint32_t VariableId;

class DifferentiableValue {
private:
  std::vector<TFloat> derivatives;
  TFloat _value;

  void align_size(const DifferentiableValue &other);

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
