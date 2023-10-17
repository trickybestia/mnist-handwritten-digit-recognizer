#include <fstream>

#include "utils.hpp"

using namespace std;
using namespace std::filesystem;

void save_parameters(const Matrix &parameters, path path) {
  ofstream file(path);

  file.write(reinterpret_cast<const char *>(parameters.data()),
             parameters.size() * sizeof(TFloat));
}

void load_parameters(Matrix &parameters, path path) {
  ifstream file(path);

  file.read(reinterpret_cast<char *>(parameters.data()),
            parameters.size() * sizeof(TFloat));
}