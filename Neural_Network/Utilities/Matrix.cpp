#include "Matrix.h"


Matrix::Matrix() {}
Matrix::Matrix(size_t row, size_t columns) : _rows(row), _cols(columns), dmatrix(row, dvector(columns, 0.0)) {}
Matrix::Matrix(std::initializer_list<std::initializer_list<double>> init) {
	*this = init;
}
Matrix::Matrix(dmatrix init) {

	this->clear();
	for (auto& row : init)
		this->emplace_back(row);

	_rows = this->size();
	_cols = (*this)[0].size();
}

void Matrix::operator=(std::initializer_list<std::initializer_list<double>> init) {

	this->clear();
	for (auto& row : init)
		this->emplace_back(row);

	_rows = this->size();
	_cols = (*this)[0].size();
}
Matrix Matrix::operator*(const Matrix& B) {
	size_t n_columns = B[0].size();

	Matrix C(_rows, n_columns);
	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < n_columns; j++)
			for (size_t k = 0; k < _cols; k++)
				C[i][j] += (*this)[i][k] * B[k][j];

	return C;
}
Matrix Matrix::hadamard(const Matrix& B) {

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < _cols; j++)
			C[i][j] = (*this)[i][j] * B[i][j];

	return C;
}
Matrix Matrix::operator*(const double& a) {

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < _cols; j++)
			C[i][j] = (*this)[i][j] * a;

	return C;
}
Matrix Matrix::operator+(const Matrix& B) {

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < _cols; j++)
			C[i][j] = (*this)[i][j] + B[i][j];

	return C;
}
Matrix Matrix::operator-(const Matrix& B) {

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < _cols; j++)
			C[i][j] = (*this)[i][j] - B[i][j];

	return C;
}

Matrix Matrix::T() {
	Matrix C(_cols, _rows);
	for (size_t i = 0; i < _cols; i++)
		for (size_t j = 0; j < _rows; j++)
			C[i][j] = (*this)[j][i];

	return C;
}
Matrix Matrix::addBias() {

	Matrix C(_rows, _cols + 1);
	for (size_t i = 0; i < _rows; i++) {
		for (size_t j = 0; j < _cols; j++)
			C[i][j] = (*this)[i][j];
		C[i][_cols] = 1;
	}

	return C;
}

// add biases and then transposes
Matrix Matrix::addBias_then_T() {

	Matrix C(_cols + 1, _rows);
	for (size_t j = 0; j < _rows; j++) {
		for (size_t i = 0; i < _cols; i++)
			C[i][j] = (*this)[j][i];
		C[_cols][j] = 1;
	}

	return C;
}

// transposes and then remove biases
Matrix Matrix::T_then_removeBias() {

	Matrix C(_cols, _rows - 1);
	for (size_t i = 0; i < _cols; i++)
		for (size_t j = 0; j < _rows - 1; j++)
			C[i][j] = (*this)[j][i];

	return C;
}

Matrix Matrix::dropoutMask(const double& dropout) {
	double keep_prob = 1.0 - dropout;

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; ++i)
		for (size_t j = 0; j < _cols; ++j)
			C[i][j] = ((double)rand() / RAND_MAX > keep_prob) ? 0.0 : ((*this)[i][j] / keep_prob);

	return C;
}
Matrix Matrix::derivReLU() {

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < _cols; j++)
			C[i][j] = ((*this)[i][j] > 0.0 ? 1.0 : 0);

	return C;
}
Matrix Matrix::setMaxToOne() {

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++) {
		auto maxElement = std::max_element((*this)[i].begin(), (*this)[i].end());
		size_t maxIndex = std::distance((*this)[i].begin(), maxElement);

		C[i][maxIndex] = 1;
	}

	return C;
}