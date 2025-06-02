#include "functions.h"

namespace DL {
	class Dense_layer {
	private:
		dmatrix m_weights;
		dvector m_biases;
		dmatrix m_output;

	public:
		Dense_layer(const int& n_inputs, const int& n_neurons);
		void forward(const dmatrix& inputs);
		dmatrix output();
	};
}

class Activation {
private:
	dmatrix m_output;
public:
	Activation(const dmatrix& inputs);
	dmatrix output();
};
class Activation_Softmax {
private:
	dmatrix m_output;
public:
	Activation_Softmax(const dmatrix& inputs);
	dmatrix output();
};


class Cross_Entropy_Loss {
private:
	dvector Loss;
	double mean;
public:
	Cross_Entropy_Loss(const dmatrix& y_pred, const dvector& y_true);
	std::pair<dvector, double> getLoss();
};