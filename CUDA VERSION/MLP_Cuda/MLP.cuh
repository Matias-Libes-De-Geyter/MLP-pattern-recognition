#include "DenseBlock.cuh"
#include <fstream>
#include <sstream>

class MLP {
private:
	hyperparameters _hyperparameters;
	std::vector<DenseBlock> m_layers;
	std::vector<Matrix> m_deltas;
	std::vector<Matrix> m_grad_J;
	std::vector<Matrix> M, V;
	int L;
	int t;
public:
	MLP(const hyperparameters& hyper);
	Matrix forward(Matrix input);
	void backpropagation(Matrix input, Matrix y_hot_one);
	void Adam();
	double backwards(Matrix input, Matrix y_hot_one);
	double CELoss(Matrix y_hot_one);
	void saveWeights(const std::string& filename);
	void loadWeights(const std::string& filename);
};