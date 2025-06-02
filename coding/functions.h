#include <iostream>
#include <vector>
#include <numeric>
#include <ctime>
#include <random>
#include <string>
#include <utility>

#define EULERS_NUMBER pow((1.0 + 1.0 / 10000000.0), 10000000.0)

using dmatrix = std::vector<std::vector<double>>;
using dvector = std::vector<double>;

double random(const double& min, const double& max);
dmatrix operator*(const dmatrix& A, const dmatrix& B);
dmatrix operator+(const dmatrix& A, const dvector& B);
dvector operator+(const dvector& A, const dvector& B);
void printArray(const dmatrix& arr, const std::string& texte);
void printArray(const dvector& arr);

std::tuple<dmatrix, dvector> spiral_data(const size_t& points, const size_t& classes, const float& spread);