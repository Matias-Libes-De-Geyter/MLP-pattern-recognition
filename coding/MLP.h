#include "DenseBlock.h"

class MLP {
private:
	std::vector<DenseBlock> m_layers;
	std::vector<dmatrix> m_deltas;
	std::vector<dmatrix> m_grad_J;
	std::vector<dmatrix> M, V;
	int L;
	int t;
public:
	MLP(const dvector& hidden_layers_sizes);
	dmatrix forward(const dmatrix& input);
	void backpropagation(const dmatrix& input, const dmatrix& y_hot_one);
	void Adam();
	double backwards(const dmatrix& input, const dmatrix& y_hot_one);
};