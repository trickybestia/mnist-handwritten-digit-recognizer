#pragma once

#include <filesystem>

#include "matrix.hpp"

void save_parameters(const Vector &parameters, std::filesystem::path path);
void load_parameters(Vector &parameters, std::filesystem::path path);
