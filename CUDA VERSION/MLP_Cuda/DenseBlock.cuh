#include "functions.cuh"

class DenseBlock {
private:
	Matrix m_weights;
	Matrix m_preactivation;
	Matrix m_output;

	dvector delta;

	void activate(const Matrix& inputs);
	void softmax_activation(const Matrix& inputs);

public:
	DenseBlock(const int& n_inputs, const int& n_neurons);
	void forward(Matrix inputs, const std::string& softmax = "");
	Matrix output();
	Matrix weights();
	void setWeights(const Matrix& weights);
	Matrix preactivation();
};