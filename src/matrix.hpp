#pragma once

#include <cstddef>
#include <iterator>
#include <vector>

#include "autograd/autograd.hpp"

template <typename TElement> class Matrix {
private:
  size_t _rows, _cols;

public:
  std::vector<TElement> data;

  Matrix();
  Matrix(size_t rows, size_t cols);
  template <typename InputIt>
  Matrix(size_t rows, size_t cols, InputIt first, InputIt last);

  size_t rows() const;
  size_t cols() const;

  const TElement &operator()(size_t i, size_t j) const;
  TElement &operator()(size_t i, size_t j);

  Matrix transpose() const;
  Matrix dot(const Matrix &other) const;

  Matrix operator*(TElement factor) const;

  Matrix operator*(const Matrix &other) const;
  Matrix operator+(const Matrix &other) const;
  Matrix operator-(const Matrix &other) const;
};
