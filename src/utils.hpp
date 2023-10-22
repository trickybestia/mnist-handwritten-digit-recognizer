#pragma once

#include <filesystem>
#include <vector>

#include "matrix.hpp"

void save_matrix(const Matrix &parameters, std::filesystem::path path);
void load_matrix(Matrix &parameters, std::filesystem::path path);
