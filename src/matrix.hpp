#pragma once

#include <cstddef>
#include <vector>

typedef float TFloat;

class Matrix {
private:
  size_t _rows, _cols;

public:
  std::vector<TFloat> data;

  Matrix();
  Matrix(size_t rows, size_t cols);

  size_t rows() const;
  size_t cols() const;

  TFloat operator()(size_t i, size_t j) const;
  TFloat &operator()(size_t i, size_t j);

  Matrix transpose() const;
  Matrix dot(const Matrix &other) const;

  void zeroize();

  void operator*=(TFloat factor);

  void operator*=(const Matrix &other);
  void operator+=(const Matrix &other);
  void operator-=(const Matrix &other);

  Matrix operator*(TFloat factor) const;

  Matrix operator*(const Matrix &other) const;
  Matrix operator+(const Matrix &other) const;
  Matrix operator-(const Matrix &other) const;
};
