#include "functions.h"

class DenseBlock {
private:
	dmatrix m_weights;
	dmatrix m_preactivation;
	dmatrix m_output;

	dvector delta;
	
	void activate(const dmatrix& inputs);
	void softmax_activation(const dmatrix& inputs);

public:
	DenseBlock(const int& n_inputs, const int& n_neurons);
	void forward(const dmatrix& inputs, const std::string& softmax = "");
	dmatrix output();
	dmatrix weights();
	void setWeights(const dmatrix& weights);
	dmatrix preactivation();
};