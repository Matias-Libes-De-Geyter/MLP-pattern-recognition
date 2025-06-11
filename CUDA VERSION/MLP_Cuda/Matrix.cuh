#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <initializer_list>
#include <string>
#include <vector>
using dmatrix = std::vector<std::vector<double>>;
using dvector = std::vector<double>;


class Matrix : public std::vector<std::vector<double>> {
private:
	size_t _rows;
	size_t _cols;
public:
	Matrix();
	Matrix(size_t row, size_t columns);
	Matrix(dmatrix);
	Matrix(std::initializer_list<std::initializer_list<double>>);
	void operator=(std::initializer_list<std::initializer_list<double>>);
	Matrix operator*(const Matrix&); // GPU
	Matrix runGPUUnary(const std::string& operation, const double& a); // Uses kernels for optimization
	Matrix runGPUBinary(const std::string& operation, const Matrix& B); // Uses kernels for optimization
	Matrix runGPUUnary(const std::string& operation); // Uses kernels for optimization
	Matrix hadamard(const Matrix&); // GPU
	Matrix operator*(const double&); // GPU
	Matrix operator+(const Matrix&); // GPU
	Matrix operator-(const Matrix&); // GPU
	Matrix T(); // GPU
	Matrix addBias(); // GPU
	Matrix addBias_then_T(); // GPU
	Matrix T_then_removeBias(); // GPU
	Matrix derivReLU(); // GPU
	Matrix setMaxToOne(); // GPU
	Matrix dropoutMask(const double&); // GPU
};