#include "functions.cuh"

double random(const double& min, const double& max) {
	std::mt19937_64 rng{};
	rng.seed(std::random_device{}());
	return std::uniform_real_distribution<>{min, max}(rng);
}
Matrix hotOne(const dvector& y, const int& nElements) {

	Matrix C(y.size(), nElements);

	for (int i = 0; i < y.size(); i++)
		C[i][y[i]] = 1;

	return C;
}


// ====================================================================== CROSS ENTROPY LOSS GPU IMPLEMENTATION
__global__ void cross_entropy_kernel(const double* y_pred, const double* y_true, double* loss, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows)
        for (int j = 0; j < cols; ++j)
            if (y_true[i * cols + j] == 1.0) {
                double prob = fmax(y_pred[i * cols + j], 1e-9);
                loss[i] = -log(prob);
                break;
            }

}

std::pair<dvector, double> CELossFunction(const Matrix& y_pred, const Matrix& y_true) {
    int rows = static_cast<int>(y_pred.size());
    int cols = static_cast<int>(y_pred[0].size());

    dvector y_pred_flat(rows * cols);
    dvector y_true_flat(rows * cols);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            y_pred_flat[i * cols + j] = y_pred[i][j];
            y_true_flat[i * cols + j] = y_true[i][j];
        }

    double *d_y_pred, *d_y_true, *d_loss;
    cudaMalloc(&d_y_pred, sizeof(double) * rows * cols);
    cudaMalloc(&d_y_true, sizeof(double) * rows * cols);
    cudaMalloc(&d_loss, sizeof(double) * rows);

    cudaMemcpy(d_y_pred, y_pred_flat.data(), sizeof(double) * rows * cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_true, y_true_flat.data(), sizeof(double) * rows * cols, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (rows + blockSize - 1) / blockSize;
    cross_entropy_kernel<<<gridSize, blockSize>>>(d_y_pred, d_y_true, d_loss, rows, cols);
    cudaDeviceSynchronize();

    dvector loss(rows, 1e-9);
    cudaMemcpy(loss.data(), d_loss, sizeof(double) * rows, cudaMemcpyDeviceToHost);

    double mean_loss = 0.0;
    for (double val : loss) mean_loss += val;
    mean_loss /= rows;

    // Cleanup
    cudaFree(d_y_pred);
    cudaFree(d_y_true);
    cudaFree(d_loss);

    return { loss, mean_loss };
}