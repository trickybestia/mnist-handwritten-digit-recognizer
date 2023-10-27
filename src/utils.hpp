#pragma once

#include <filesystem>
#include <vector>

#include "matrix.hpp"

void save_matrix(const Matrix &parameters, std::filesystem::path path);
void load_matrix(Matrix &parameters, std::filesystem::path path);

size_t max_item_index(const Matrix &matrix);
