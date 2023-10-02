#include <iostream>
#include <random>

#include "matrix.hpp"

using namespace std;

Matrix::Matrix() : _rows(0), _cols(0) {}

Matrix::Matrix(size_t rows, size_t cols)
    : _rows(rows), _cols(cols), data(rows * cols) {}

size_t Matrix::rows() const { return this->_rows; }
size_t Matrix::cols() const { return this->_cols; }

void Matrix::randomize(TFloat min, TFloat max) {
  random_device rd;
  uniform_real_distribution<TFloat> distribution(min, max);

  for (size_t i = 0; i != this->data.size(); i++) {
    this->data[i] = distribution(rd);
  }
}

TFloat &Matrix::operator()(size_t i, size_t j) {
  return this->data.at(i * this->_cols + j);
}

const TFloat &Matrix::operator()(size_t i, size_t j) const {
  return this->data.at(i * this->_cols + j);
}

Matrix Matrix::transpose() const {
  Matrix result(this->_cols, this->_rows);

  for (size_t i = 0; i != this->_rows; i++) {
    for (size_t j = 0; j != this->_cols; j++) {
      result(j, i) = (*this)(i, j);
    }
  }

  return result;
}

Matrix Matrix::dot(const Matrix &other) const {
  if (this->_cols != other._rows)
    throw exception();

  Matrix result(this->_rows, other._cols);

  for (size_t i = 0; i != this->_rows; i++) {
    for (size_t j = 0; j != other._cols; j++) {
      result(i, j) = 0.0;

      for (size_t k = 0; k != this->_cols; k++) {
        result(i, j) += (*this)(i, k) * other(k, j);
      }
    }
  }

  return result;
}

Matrix Matrix::operator*(TFloat factor) const {
  Matrix result(this->_rows, this->_cols);

  for (size_t i = 0; i != this->data.size(); i++) {
    result.data[i] = this->data[i] * factor;
  }

  return result;
}

Matrix Matrix::operator+(const Matrix &other) const {
  if (this->_cols != other._cols || this->_rows != other._rows)
    throw exception();

  Matrix result(this->_rows, this->_cols);

  for (size_t i = 0; i != this->data.size(); i++) {
    result.data[i] = this->data[i] + other.data[i];
  }

  return result;
}

Matrix Matrix::operator-(const Matrix &other) const {
  if (this->_cols != other._cols || this->_rows != other._rows)
    throw exception();

  Matrix result(this->_rows, this->_cols);

  for (size_t i = 0; i != this->data.size(); i++) {
    result.data[i] = this->data[i] - other.data[i];
  }

  return result;
}

Matrix Matrix::operator*(const Matrix &other) const {
  if (this->_cols != other._cols || this->_rows != other._rows)
    throw exception();

  Matrix result(this->_rows, this->_cols);

  for (size_t i = 0; i != this->data.size(); i++) {
    result.data[i] = this->data[i] * other.data[i];
  }

  return result;
}
