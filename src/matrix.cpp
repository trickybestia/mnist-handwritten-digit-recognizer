#include <iomanip>
#include <iostream>
#include <random>

#include "matrix.hpp"

using namespace std;

Matrix::Matrix() : _rows(0), _cols(0) {}

Matrix::Matrix(size_t rows, size_t cols)
    : _rows(rows), _cols(cols), data(rows * cols) {}

size_t Matrix::rows() const { return this->_rows; }
size_t Matrix::cols() const { return this->_cols; }

TFloat &Matrix::operator()(size_t i, size_t j) {
  return this->data.at(i * this->_cols + j);
}

TFloat Matrix::operator()(size_t i, size_t j) const {
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
      result(i, j) = (*this)(i, 0) * other(0, j);

      for (size_t k = 1; k != this->_cols; k++) {
        result(i, j) += (*this)(i, k) * other(k, j);
      }
    }
  }

  return result;
}

void Matrix::zeroize() {
  for (size_t i = 0; i != this->data.size(); i++)
    this->data[i] = 0.0;
}

void Matrix::operator*=(TFloat factor) {
  for (size_t i = 0; i != this->data.size(); i++) {
    this->data[i] *= factor;
  }
}

void Matrix::operator+=(const Matrix &other) {
  if (this->_cols != other._cols || this->_rows != other._rows)
    throw exception();

  for (size_t i = 0; i != this->data.size(); i++) {
    this->data[i] += other.data[i];
  }
}

void Matrix::operator-=(const Matrix &other) {
  if (this->_cols != other._cols || this->_rows != other._rows)
    throw exception();

  for (size_t i = 0; i != this->data.size(); i++) {
    this->data[i] -= other.data[i];
  }
}

void Matrix::operator*=(const Matrix &other) {
  if (this->_cols != other._cols || this->_rows != other._rows)
    throw exception();

  for (size_t i = 0; i != this->data.size(); i++) {
    this->data[i] *= other.data[i];
  }
}

Matrix Matrix::operator*(TFloat factor) const {
  Matrix result = *this;

  result *= factor;

  return result;
}

Matrix Matrix::operator+(const Matrix &other) const {
  Matrix result = *this;

  result += other;

  return result;
}

Matrix Matrix::operator-(const Matrix &other) const {
  Matrix result = *this;

  result -= other;

  return result;
}

Matrix Matrix::operator*(const Matrix &other) const {
  Matrix result = *this;

  result *= other;

  return result;
}
