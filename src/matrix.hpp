#pragma once

#include <cstddef>
#include <vector>

#include "float.hpp"

class Matrix {
private:
  size_t _rows, _cols;

public:
  std::vector<TFloat> data;

  Matrix();

  Matrix(size_t rows, size_t cols);

  size_t rows() const;
  size_t cols() const;

  void randomize(TFloat min, TFloat max);

  const TFloat &operator()(size_t i, size_t j) const;
  TFloat &operator()(size_t i, size_t j);

  Matrix transpose() const;
  Matrix dot(const Matrix &other) const;

  Matrix operator*(TFloat factor) const;

  Matrix operator*(const Matrix &other) const;
  Matrix operator+(const Matrix &other) const;
  Matrix operator-(const Matrix &other) const;
};
