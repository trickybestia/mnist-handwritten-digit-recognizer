#include <iostream>
#include <random>

#include "matrix.hpp"

using namespace std;

template <typename TElement> Matrix<TElement>::Matrix() : _rows(0), _cols(0) {}

template <typename TElement>
Matrix<TElement>::Matrix(size_t rows, size_t cols)
    : _rows(rows), _cols(cols), data(rows * cols) {}

template <typename TElement> size_t Matrix<TElement>::rows() const {
  return this->_rows;
}
template <typename TElement> size_t Matrix<TElement>::cols() const {
  return this->_cols;
}

template <typename TElement>
TElement &Matrix<TElement>::operator()(size_t i, size_t j) {
  return this->data.at(i * this->_cols + j);
}

template <typename TElement>
const TElement &Matrix<TElement>::operator()(size_t i, size_t j) const {
  return this->data.at(i * this->_cols + j);
}

template <typename TElement>
Matrix<TElement> Matrix<TElement>::transpose() const {
  Matrix<TElement> result(this->_cols, this->_rows);

  for (size_t i = 0; i != this->_rows; i++) {
    for (size_t j = 0; j != this->_cols; j++) {
      result(j, i) = (*this)(i, j);
    }
  }

  return result;
}

template <typename TElement>
Matrix<TElement> Matrix<TElement>::dot(const Matrix<TElement> &other) const {
  if (this->_cols != other._rows)
    throw exception();

  Matrix<TElement> result(this->_rows, other._cols);

  for (size_t i = 0; i != this->_rows; i++) {
    for (size_t j = 0; j != other._cols; j++) {
      result(i, j) = (*this)(i, 0) * other(0, j);

      for (size_t k = 1; k != this->_cols; k++) {
        result(i, j) += (*this)(i, k) * other(k, j);
      }
    }
  }

  return result;
}

template <typename TElement>
void Matrix<TElement>::operator*=(TElement factor) {
  for (size_t i = 0; i != this->data.size(); i++) {
    this->data[i] *= factor;
  }
}

template <typename TElement>
void Matrix<TElement>::operator+=(const Matrix<TElement> &other) {
  if (this->_cols != other._cols || this->_rows != other._rows)
    throw exception();

  for (size_t i = 0; i != this->data.size(); i++) {
    this->data[i] += other.data[i];
  }
}

template <typename TElement>
void Matrix<TElement>::operator-=(const Matrix<TElement> &other) {
  if (this->_cols != other._cols || this->_rows != other._rows)
    throw exception();

  for (size_t i = 0; i != this->data.size(); i++) {
    this->data[i] -= other.data[i];
  }
}

template <typename TElement>
void Matrix<TElement>::operator*=(const Matrix<TElement> &other) {
  if (this->_cols != other._cols || this->_rows != other._rows)
    throw exception();

  for (size_t i = 0; i != this->data.size(); i++) {
    this->data[i] *= other.data[i];
  }
}

template <typename TElement>
Matrix<TElement> Matrix<TElement>::operator*(TElement factor) const {
  Matrix<TElement> result = *this;

  result *= factor;

  return result;
}

template <typename TElement>
Matrix<TElement>
Matrix<TElement>::operator+(const Matrix<TElement> &other) const {
  Matrix<TElement> result = *this;

  result += other;

  return result;
}

template <typename TElement>
Matrix<TElement>
Matrix<TElement>::operator-(const Matrix<TElement> &other) const {
  Matrix<TElement> result = *this;

  result -= other;

  return result;
}

template <typename TElement>
Matrix<TElement>
Matrix<TElement>::operator*(const Matrix<TElement> &other) const {
  Matrix<TElement> result = *this;

  result *= other;

  return result;
}

template class Matrix<TFloat>;
