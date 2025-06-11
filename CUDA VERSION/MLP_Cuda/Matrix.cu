#include "Matrix.cuh"
#include <curand_kernel.h>

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

// ================================================================================================================== MULTIPLICATION
__global__ void multKernel(double* A, double* B, double* C, int A_rows, int A_cols, int B_cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < A_rows && col < B_cols) {
		double sum = 0.0;
		for (int k = 0; k < A_cols; k++) {
			sum += A[row * A_cols + k] * B[k * B_cols + col];
		}
		C[row * B_cols + col] = sum;
	}
}
Matrix Matrix::operator*(const Matrix& B) {
	size_t B_cols = B[0].size();
	Matrix C(_rows, B_cols);

	// Device allocs
	double* d_A, * d_B, * d_C;
	size_t size_A = _rows * _cols * sizeof(double);
	size_t size_B = _cols * B_cols * sizeof(double);
	size_t size_C = _rows * B_cols * sizeof(double);
	cudaMalloc(&d_A, size_A);
	cudaMalloc(&d_B, size_B);
	cudaMalloc(&d_C, size_C);

	// Flatten matrices (switch to vector later on)
	std::vector<double> flat_A, flat_B;
	for (auto& row : *this) flat_A.insert(flat_A.end(), row.begin(), row.end());
	for (auto& row : B) flat_B.insert(flat_B.end(), row.begin(), row.end());

	// Memory copy from host to device
	cudaMemcpy(d_A, flat_A.data(), size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, flat_B.data(), size_B, cudaMemcpyHostToDevice);

	// Spit dimensions
	dim3 threadsPerBlock(16, 16); dim3 blocksPerGrid((B_cols + 15) / 16, (_rows + 15) / 16);
	multKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, _rows, _cols, B_cols); // Make operation

	// Paste operation result to the host
	std::vector<double> flat_C(_rows * B_cols);
	cudaMemcpy(flat_C.data(), d_C, size_C, cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < _rows; ++i)
		for (size_t j = 0; j < B_cols; ++j)
			C[i][j] = flat_C[i * B_cols + j];

	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); // Free allocs
	return C;
}


// ================================= HADAMARD
__global__ void hadamardKernel(const double* A, const double* B, double* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] * B[idx];
}
Matrix Matrix::hadamard(const Matrix& B) {
	return runGPUBinary("hadamardKernel", B);
}

// ================================= SCALAR MULTIPLICATION
__global__ void scalarMulKernel(const double* A, double* C, double a, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) C[idx] = A[idx] * a;
}
Matrix Matrix::operator*(const double& a) {
	return runGPUUnary("scalar multiplication", a);
}


// ================================= ADDITION
__global__ void addKernel(const double* A, const double* B, double* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) C[idx] = A[idx] + B[idx];
}
Matrix Matrix::operator+(const Matrix& B) {
	return runGPUBinary("addKernel", B);
}

// ================================= SUBSTRACTION
__global__ void subKernel(const double* A, const double* B, double* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) C[idx] = A[idx] - B[idx];
}
Matrix Matrix::operator-(const Matrix& B) {
	return runGPUBinary("subKernel", B);
}

// ================================= TRANSPOSITION
__global__ void transposeKernel(const double* A, double* C, int rows, int cols) {
	int i = blockDim.y * blockIdx.y + threadIdx.y; // row
	int j = blockIdx.x * blockDim.x + threadIdx.x; // col

	if (i < rows && j < cols)
		C[j * rows + i] = A[i * cols + j];
}
Matrix Matrix::T() {
	return runGPUUnary("transposition");
}

// ================================= ADD BIASES
__global__ void addBiasKernel(const double* A, double* C, int rows, int cols) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < rows) {
		for (int j = 0; j < cols; ++j)
			C[i * (cols + 1) + j] = A[i * cols + j];

		C[i * (cols + 1) + cols] = 1.0;
	}
}
Matrix Matrix::addBias() {
	return runGPUUnary("add bias");
}

// ================================= ADD BIASES AND THEN TRANSPOSE
__global__ void addBiasAndTKernel(const double* A, double* C, int rows, int cols) {
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < cols && j < rows)
		C[i * rows + j] = A[j * cols + i];
	if (i == cols && j < rows)
		C[cols * rows + j] = 1.0;
}
Matrix Matrix::addBias_then_T() {
	return runGPUUnary("add bias then transpose");
}

// ================================= TRANSPOSE THEN REMOVE BIASES
__global__ void TAndRemoveBiasKernel(const double* A, double* C, int rows, int cols) {
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < cols && j < rows - 1)
		C[i * (rows - 1) + j] = A[j * cols + i];
}
Matrix Matrix::T_then_removeBias() {
	return runGPUUnary("transpose and remove bias");
}



// ================================= DROPOUT
__global__ void dropoutKernel(const double* A, double* C, double keep_prob, int size, unsigned long seed) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;

	curandState state;
	curand_init(seed, idx, 0, &state);

	double randVal = curand_uniform_double(&state);
	C[idx] = (randVal > keep_prob) ? 0.0 : (A[idx] / keep_prob);
}
Matrix Matrix::dropoutMask(const double& dropout) {
	return runGPUUnary("dropout", dropout);
}

// ================================= APPLY ReLU DERIVATE
__global__ void derivReLUKernel(const double* A, double* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		C[idx] = (A[idx] > 0.0) ? 1.0 : 0.0;
}
Matrix Matrix::derivReLU() {
	return runGPUUnary("derivate of ReLU");
}


// Set Maximum of vectors to 1 and others to 0.
__global__ void getMaxKernel(const double* A, double* C, int rows, int cols) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < rows) {
		double maxVal = A[i * cols];
		int maxIdx = 0;

		for (int j = 1; j < cols; ++j) {
			double val = A[i * cols + j];
			if (val > maxVal) {
				maxVal = val;
				maxIdx = j;
			}
		}

		for (int j = 0; j < cols; ++j)
			C[i * cols + j] = (j == maxIdx) ? 1.0 : 0.0;
	}

}

Matrix Matrix::setMaxToOne() {
	return runGPUUnary("set max to one and other to zero");
}


Matrix Matrix::runGPUBinary(const std::string& operation, const Matrix& B) {
	int N = _rows * _cols;
	Matrix C(_rows, _cols);

	// device allocations
	double* d_A, * d_B, * d_C;
	size_t bytes = N * sizeof(double);
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);

	// flat copy (may implement Matrices as Vectors later on
	std::vector<double> flat_A; flat_A.reserve(N);
	for (auto& row : *this)
		flat_A.insert(flat_A.end(), row.begin(), row.end());
	std::vector<double> flat_B; flat_B.reserve(N);
	for (auto& row : B)
		flat_B.insert(flat_B.end(), row.begin(), row.end());

	// Memory Copy from Host to Device
	cudaMemcpy(d_A, flat_A.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, flat_B.data(), bytes, cudaMemcpyHostToDevice);

	// Kernel launch in 1D
	int TPB = 256;
	int nBlocks = (N + TPB - 1) / TPB;
	if (operation == "hadamardKernel")
		hadamardKernel<<<nBlocks, TPB>>>(d_A, d_B, d_C, N);
	if (operation == "addKernel")
		addKernel<<<nBlocks, TPB>>>(d_A, d_B, d_C, N);
	if (operation == "subKernel")
		subKernel<<<nBlocks, TPB>>>(d_A, d_B, d_C, N);

	// Get the flat vector out of the kernel
	std::vector<double> flat_C(N);
	cudaMemcpy(flat_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < _rows; ++i)
		for (size_t j = 0; j < _cols; ++j)
			C[i][j] = flat_C[i * _cols + j];

	// Free allocations
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return C;
}

Matrix Matrix::runGPUUnary(const std::string& operation, const double& a) {
	int C_rows = _rows;
	int C_cols = _cols;
	if (operation == "add bias")
		C_cols = _cols + 1;
	if (operation == "transposition") {
		C_rows = _cols;
		C_cols = _rows;
	}
	if (operation == "add bias then transpose") {
		C_rows = _cols + 1;
		C_cols = _rows;
	}
	if (operation == "transpose and remove bias") {
		C_rows = _cols;
		C_cols = _rows - 1;
	}

	int N = C_rows * C_cols;
	Matrix C(C_rows, C_cols);

	double* d_A, * d_C;
	size_t size_A = _rows * _cols * sizeof(double);
	size_t size_C = N * sizeof(double);

	cudaMalloc(&d_A, size_A);
	cudaMalloc(&d_C, size_C);

	std::vector<double> flat_A; flat_A.reserve(N);
	for (auto& row : *this) flat_A.insert(flat_A.end(), row.begin(), row.end());

	cudaMemcpy(d_A, flat_A.data(), size_A, cudaMemcpyHostToDevice);

	// Thread and block sizes for 2D and 1D Kernel.
	int TPB = 256;
	int nBlocks = (N + TPB - 1) / TPB;

	dim3 blockSize(16, 16);
	dim3 gridSize((C_cols + blockSize.x - 1) / blockSize.x,
				  (C_rows + blockSize.y - 1) / blockSize.y);
	if(operation == "scalar multiplication")
		scalarMulKernel<<<nBlocks, TPB>>>(d_A, d_C, a, _rows * _cols);
	if(operation == "transposition")
		transposeKernel<<<gridSize, blockSize>>>(d_A, d_C, _rows, _cols);
	if(operation == "add bias")
		addBiasKernel<<<nBlocks, TPB>>>(d_A, d_C, _rows, _cols);
	if(operation == "add bias then transpose")
		addBiasAndTKernel<<<gridSize, blockSize>>>(d_A, d_C, _rows, _cols);
	if(operation == "transpose and remove bias")
		TAndRemoveBiasKernel<<<gridSize, blockSize>>>(d_A, d_C, _rows, _cols);
	if (operation == "dropout")
		dropoutKernel<<<nBlocks, TPB>>>(d_A, d_C, 1.0 - a, _rows * _cols, time(NULL));
	if (operation == "derivate of ReLU")
		derivReLUKernel<<<nBlocks, TPB>>>(d_A, d_C, _rows * _cols);
	if (operation == "set max to one and other to zero")
		getMaxKernel<<<nBlocks, TPB>>>(d_A, d_C, _rows, _cols);

	// Copy result back
	std::vector<double> flat_C(C_rows * C_cols);
	cudaMemcpy(flat_C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

	// Fill C matrix
	for (size_t i = 0; i < C_rows; ++i)
		for (size_t j = 0; j < C_cols; ++j)
			C[i][j] = flat_C[i * C_cols + j];

	// Cleanup
	cudaFree(d_A); cudaFree(d_C);

	return C;
}

Matrix Matrix::runGPUUnary(const std::string& operation) {
	return runGPUUnary(operation, 0.0);
}