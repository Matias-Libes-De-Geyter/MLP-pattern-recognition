#include <iostream>
#include <vector>
#include <numeric>
#include <ctime>
#include <cmath>
#include <random>
#include <string>
#include <utility>
#include <algorithm>
#include <fstream>

#define EULERS_NUMBER pow((1.0 + 1.0 / 10000000.0), 10000000.0)

using dmatrix = std::vector<std::vector<double>>;
using dvector = std::vector<double>;

double random(const double& min, const double& max);
dmatrix operator*(const dmatrix& A, const dmatrix& B);
dmatrix operator*(const double& a, const dmatrix& B);
dmatrix aug_inputs_mult(const dmatrix& A, const dmatrix& B);
dmatrix operator+(const dmatrix& A, const dmatrix& B);
dmatrix operator+(const dmatrix& A, const dvector& B);
dvector operator+(const dvector& A, const dvector& B);
dmatrix operator-(const dmatrix& A, const dmatrix& B);
dmatrix addBiases(const dmatrix& A);
dmatrix transpose(const dmatrix& A);
dmatrix remove_bias(const dmatrix& A);
dmatrix hadamard(const dmatrix& A, const dmatrix& B);
dmatrix ReLU_derivate(const dmatrix& A);
dmatrix getCertitudeHot(const dmatrix& A);
dmatrix getCertitude(const dmatrix& A);
std::pair<dvector, double> CELoss(const dmatrix& y_pred, const dmatrix& y_true);

void readMNIST(const std::string& imageFile, const std::string& labelFile, dmatrix& images, dvector& labels);

void print(const dmatrix& arr, const std::string& texte);
void print(const dmatrix& arr);
void print(const dvector& arr);
void print(const double& val);
void print(const std::string& texte);
void print(const double& val, const std::string& texte);
void printSize(const dmatrix& mat, int l, std::string type);

std::tuple<dmatrix, dmatrix> spiral_data(const size_t& points, const size_t& classes, const float& spread);
dmatrix hotOne(const dvector& y, const int& nElements);