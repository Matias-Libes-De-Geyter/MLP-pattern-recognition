#include "Blocks.h"


// ======== DENSE LAYER ======== //
DL::Dense_layer::Dense_layer(const int& n_inputs, const int& n_neurons) : m_weights(n_inputs, dvector(n_neurons)), m_biases(n_neurons, 0) {
	for (size_t i = 0; i < n_neurons; i++) {
		for (size_t j = 0; j < n_inputs; j++) {
			m_weights[j][i] = random(-1, 1);
		}
		m_biases[i] = 0;//random(0, 1);
	}
};
void DL::Dense_layer::forward(const dmatrix& inputs) {
	m_output = inputs * m_weights + m_biases;
};
dmatrix DL::Dense_layer::output() {
	return m_output;
}


// ======== ACTIVATION RELU ======== //
Activation::Activation(const dmatrix& inputs) : m_output(inputs.size(), dvector(inputs[0].size())) {
	for (int i = 0; i < inputs.size(); i++) {
		for (int j = 0; j < inputs[0].size(); j++) {
			m_output[i][j] = std::max(0.0, inputs[i][j]);
		}
	}
}
dmatrix Activation::output() {
	return m_output;
}

// ======== ACTIVATION SOFTMAX ======== //
Activation_Softmax::Activation_Softmax(const dmatrix& inputs) : m_output(inputs.size(), dvector(inputs[0].size(), 0)) {

	dvector maxs(inputs.size());
	for (int i = 0; i < inputs.size(); i++) {
		maxs[i] = inputs[i][0];
		for (int j = 0; j < inputs[0].size(); j++) {
			if (inputs[i][j] > maxs[i]) {
				maxs[i] = inputs[i][j];
			}
		}
	}

	dmatrix expvalues(inputs);
	dvector sum_of_exps(inputs.size(), 0);
	for (int i = 0; i < inputs.size(); i++) {
		for (int j = 0; j < inputs[0].size(); j++) {
			expvalues[i][j] = pow(EULERS_NUMBER, inputs[i][j] - maxs[i]);
			sum_of_exps[i] += expvalues[i][j];
		}
	}

	for (int i = 0; i < inputs.size(); i++) {
		for (int j = 0; j < inputs[0].size(); j++) {
			m_output[i][j] = expvalues[i][j] / sum_of_exps[i]; // m_output = probabilities
		}
	}

}
dmatrix Activation_Softmax::output() {
	return m_output;
}




Cross_Entropy_Loss::Cross_Entropy_Loss(const dmatrix& y_pred, const dvector& y_true) : Loss(y_pred.size(), 0), mean(0) {
	for (int i = 0; i < y_pred.size(); i++) {
		Loss[i] = y_pred[i][y_true[i]];
		mean += Loss[i];
	}
	mean /= y_pred.size();
}
std::pair<dvector, double> Cross_Entropy_Loss::getLoss() {
	return {Loss, mean};
}