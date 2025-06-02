#include "functions.h"

class DenseBlock {
private:
	dmatrix m_weights;
	dvector m_biases;
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
	dmatrix preactivation();
};


class Cross_Entropy_Loss {
private:
	dvector Loss;
	double mean;
public:
	Cross_Entropy_Loss(const dmatrix& y_pred, const dvector& y_true);
	std::pair<dvector, double> getLoss();
};