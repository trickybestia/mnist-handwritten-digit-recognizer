#pragma once

#include <cstddef>
#include <vector>

template <typename TElement> class Matrix {
private:
  size_t _rows, _cols;

public:
  std::vector<TElement> data;

  Matrix();
  Matrix(size_t rows, size_t cols);

  size_t rows() const;
  size_t cols() const;

  const TElement &operator()(size_t i, size_t j) const;
  TElement &operator()(size_t i, size_t j);

  Matrix transpose() const;
  Matrix dot(const Matrix &other) const;

  void operator*=(TElement factor);

  void operator*=(const Matrix &other);
  void operator+=(const Matrix &other);
  void operator-=(const Matrix &other);

  Matrix operator*(TElement factor) const;

  Matrix operator*(const Matrix &other) const;
  Matrix operator+(const Matrix &other) const;
  Matrix operator-(const Matrix &other) const;
};
