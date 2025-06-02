#include "functions.h"

double random(const double& min, const double& max) {
	std::mt19937_64 rng{};
	rng.seed(std::random_device{}());
	return std::uniform_real_distribution<>{min, max}(rng);
}

dmatrix operator*(const dmatrix& A, const dmatrix& B) {
	size_t n_rows = A.size();
	size_t n_columns = B[0].size();
	size_t n_inner = B.size();

	dmatrix C(n_rows, dvector(n_columns, 0));
	for (size_t i = 0; i < n_rows; i++) {
		for (size_t j = 0; j < n_columns; j++) {
			for (size_t k = 0; k < n_inner; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}

	return C;
}

dmatrix operator+(const dmatrix& A, const dvector& B) {
	size_t n_rows = A.size();
	size_t n_columns = A[0].size();

	dmatrix C(n_rows, dvector(n_columns, 0));
	for (size_t i = 0; i < n_rows; i++) {
		for (size_t j = 0; j < n_columns; j++) {
			C[i][j] += A[i][j] + B[j];
		}
	}

	return C;
}
dvector operator+(const dvector& A, const dvector& B) {
	size_t n_rows = A.size();

	dvector C(n_rows, 0);
	for (size_t i = 0; i < n_rows; i++) {
		C[i] += A[i] + B[i];
	}

	return C;
}
dmatrix operator-(const dmatrix& A, const dmatrix& B) {
	size_t n_rows = A.size();
	size_t n_columns = A[0].size();

	dmatrix C(n_rows, dvector(n_columns, 0));
	for (size_t i = 0; i < n_rows; i++) {
		for (size_t j = 0; j < n_columns; j++) {
			C[i][j] += A[i][j] - B[i][j];
		}
	}

	return C;
}
dmatrix transpose(const dmatrix& A) {
	size_t n_rows = A.size();
	size_t n_columns = A[0].size();

	dmatrix C(n_rows, dvector(n_columns, 0));
	for (size_t i = 0; i < n_rows; i++) {
		for (size_t j = 0; j < n_columns; j++) {
			C[i][j] = A[j][i];
		}
	}

	return C;
}
dmatrix hadamard(const dmatrix& A, const dmatrix& B) {
	size_t n_rows = A.size();
	size_t n_columns = A[0].size();

	dmatrix C(n_rows, dvector(n_columns, 0));
	for (size_t i = 0; i < n_rows; i++) {
		for (size_t j = 0; j < n_columns; j++) {
			C[i][j] += A[i][j]*B[i][j];
		}
	}

	return C;
}
dmatrix ReLU_derivate(const dmatrix& A) {
	size_t n_rows = A.size();
	size_t n_columns = A[0].size();

	dmatrix C(n_rows, dvector(n_columns, 0));
	for (size_t i = 0; i < n_rows; i++) {
		for (size_t j = 0; j < n_columns; j++) {
			C[i][j] = (A[i][j] > 0.0 ? 1.0 : 0);
		}
	}
	return C;
}


void printArray(const dmatrix& arr, const std::string& texte) {
	std::cout << texte << std::endl << "[";
	for (const auto& row : arr) {
		std::cout << "[";
		for (const auto& value : row) {
			std::cout << value << ", ";
		}
		std::cout << "]," << std::endl;
	}
	std::cout << "]" << std::endl;
}
void printArray(const dvector& arr) {
	std::cout << "[";
	for (const auto& value : arr) {
		std::cout << value << ", ";
	}
	std::cout << "]" << std::endl;
}


std::tuple<std::tuple<dmatrix, dmatrix>, dvector> spiral_data(const size_t& points, const size_t& classes, const float& spread) {
	dmatrix X(points * classes, dvector(2, 0));
	dvector y(points * classes, 0);
	dmatrix y_hot_one(points * classes, dvector(classes, 0));
	double r, t;
	for (size_t i = 0; i < classes; i++) {
		for (size_t j = 0; j < points; j++) {
			r = double(j) / double(points);
			t = i * 4 + (4 * r);
			X[i * points + j] = dvector{ r * cos(t * 2.5), r * sin(t * 2.5) } + dvector{ random(-spread, spread), random(-spread, spread) };
			dvector rez;
			for (int k = 0; k < classes; k++)
				rez.push_back((i == k ? 1 : 0));
			y_hot_one[i * points + j] = rez;
			y[i * points + j] = i;
		}
	}
	return std::make_tuple(std::make_tuple(X, y_hot_one), y);
}