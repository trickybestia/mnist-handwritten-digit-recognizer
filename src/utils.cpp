#include <fstream>

#include "utils.hpp"

using namespace std;
using namespace std::filesystem;

void save_matrix(const Matrix &matrix, path path) {
  ofstream file(path);

  file.write(reinterpret_cast<const char *>(matrix.data()),
             matrix.size() * sizeof(TFloat));
}

void load_matrix(Matrix &matrix, path path) {
  ifstream file(path);

  file.read(reinterpret_cast<char *>(matrix.data()),
            matrix.size() * sizeof(TFloat));
}

size_t max_item_index(const Matrix &matrix) {
  size_t result = 0;

  for (size_t i = 0; i != matrix.size(); i++) {
    if (matrix(i) > matrix(result)) {
      result = i;
    }
  }

  return result;
}
