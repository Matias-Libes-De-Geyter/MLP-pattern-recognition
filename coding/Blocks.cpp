#include "Blocks.h"


// ======== DENSE LAYER ======== //
DenseBlock::DenseBlock(const int& n_inputs, const int& n_neurons) : m_weights(n_inputs, dvector(n_neurons)), m_biases(n_neurons, 0) {
	for (size_t i = 0; i < n_neurons; i++) {
		for (size_t j = 0; j < n_inputs; j++) {
			m_weights[j][i] = random(-1, 1);
		}
		m_biases[i] = random(-1, 1);
	}
};

void DenseBlock::activate(const dmatrix& inputs) {
	for (int i = 0; i < inputs.size(); i++) {
		for (int j = 0; j < inputs[0].size(); j++) {
			m_output[i][j] = std::max(0.0, inputs[i][j]);
		}
	}
}
void DenseBlock::softmax_activation(const dmatrix& inputs) {
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
void DenseBlock::forward(const dmatrix& inputs, const std::string& softmax) {
	m_preactivation = inputs * m_weights + m_biases;
	m_output = m_preactivation;

	(softmax == "softmax" ? softmax_activation(m_output) : activate(m_output));
};

dmatrix DenseBlock::output() {
	return m_output;
}
dmatrix DenseBlock::weights() {
	return m_weights;
}
dmatrix DenseBlock::preactivation() {
	return m_preactivation;
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