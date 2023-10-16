#pragma once

#include <filesystem>
#include <vector>

#include "matrix.hpp"

void save_parameters(const Matrix &parameters, std::filesystem::path path);
void load_parameters(Matrix &parameters, std::filesystem::path path);
