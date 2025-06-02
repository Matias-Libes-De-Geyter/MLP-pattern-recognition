	#include "DenseBlock.h"


	// ======== DENSE LAYER ======== //
	DenseBlock::DenseBlock(const int& n_inputs, const int& n_neurons) : m_weights(n_inputs + 1, dvector(n_neurons)) {

		double limit = std::sqrt(6.0 / (n_inputs + n_neurons));

		for (size_t i = 0; i < n_neurons; i++) {
			for (size_t j = 0; j < n_inputs; j++) {
				m_weights[j][i] = random(-limit, limit);
				//m_weights[j][i] = random(-1, 1);
			}
			m_weights[n_inputs][i] = random(-limit, limit);
			//m_weights[n_inputs][i] = 0.0;// random(-1, 1);
		}
	};

	void DenseBlock::activate(const dmatrix& inputs) {
		for (int i = 0; i < inputs.size(); i++)
			for (int j = 0; j < inputs[0].size(); j++)
				m_output[i][j] = std::max(0.0, inputs[i][j]);
	}


	void DenseBlock::softmax_activation(const dmatrix& inputs) {

		dvector maxs(inputs.size());
		for (int i = 0; i < inputs.size(); i++) {
			maxs[i] = inputs[i][0];
			for (int j = 0; j < inputs[0].size(); j++)
				if (inputs[i][j] > maxs[i])
					maxs[i] = inputs[i][j];
		}

		dmatrix expvalues(inputs);
		dvector sum_of_exps(inputs.size(), 0);
		for (int i = 0; i < inputs.size(); i++) {
			for (int j = 0; j < inputs[0].size(); j++) {
				expvalues[i][j] = pow(EULERS_NUMBER, inputs[i][j] - maxs[i]);
				sum_of_exps[i] += expvalues[i][j];
			}
		}

		for (int i = 0; i < inputs.size(); i++)
			for (int j = 0; j < inputs[0].size(); j++)
				m_output[i][j] = expvalues[i][j] / sum_of_exps[i]; // m_output = probabilities

	}


	void DenseBlock::forward(const dmatrix& inputs, const std::string& softmax) {
		m_preactivation = addBiases(inputs) * m_weights;

		m_output = m_preactivation;
		(softmax == "softmax" ? softmax_activation(m_output) : activate(m_output));
	};

	dmatrix DenseBlock::output() {
		return m_output;
	}
	dmatrix DenseBlock::weights() {
		return m_weights;
	}
	void DenseBlock::setWeights(const dmatrix& weights) {
		m_weights = weights;
	}
	dmatrix DenseBlock::preactivation() {
		return m_preactivation;
	}