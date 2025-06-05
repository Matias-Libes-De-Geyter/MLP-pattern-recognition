#include "MLP.h"

MLP::MLP(const hyperparameters& hyper) : _hyperparameters(hyper), L(_hyperparameters.hidden_layers_sizes.size() + 1) {

    std::vector<Matrix> matrix_L_sized(L);
    m_deltas = matrix_L_sized;
    m_grad_J = matrix_L_sized;

    dvector hidden_layers_sizes = _hyperparameters.hidden_layers_sizes;
    hidden_layers_sizes.insert(hidden_layers_sizes.begin(), _hyperparameters.input_dim);
    hidden_layers_sizes.push_back(_hyperparameters.output_dim);
    for (int l = 0; l < L; l++) {
        DenseBlock layer(hidden_layers_sizes[l], hidden_layers_sizes[l + 1]);
        m_layers.push_back(layer);

        M.push_back(Matrix(hidden_layers_sizes[l] + 1, hidden_layers_sizes[l + 1]));
        V.push_back(M[l]);
    }
    t = 0;

};

Matrix MLP::forward(Matrix input) {

    m_layers[0].forward(input);
    for (int l = 1; l < L - 1; l++)
        m_layers[l].forward(_hyperparameters.learn ? m_layers[l - 1].output().dropoutMask(_hyperparameters.dropout_rate) : m_layers[l - 1].output());
    m_layers[L - 1].forward(m_layers[L - 2].output(), "softmax");

    return m_layers.back().output();

}

void MLP::backpropagation(Matrix input, Matrix y_hot_one) {

    m_deltas[L - 1] = m_layers.back().output() - y_hot_one; // 100% sur que les deux sont de dimension 15x3
    m_grad_J[L - 1] = m_layers[L - 2].output().addBias_then_T() * m_deltas[L - 1];

    for (int l = L - 2; l >= 0; l--) {
        m_deltas[l] = (m_deltas[l + 1] * m_layers[l + 1].weights().T_then_removeBias()).hadamard(m_layers[l].preactivation().derivReLU());
        m_grad_J[l] = (l == 0 ? input : m_layers[l - 1].output()).addBias_then_T() * m_deltas[l];
        //important addBiases
    }

}

void MLP::Adam() {
    double beta_m = _hyperparameters.adam_beta_m;
    double beta_v = _hyperparameters.adam_beta_v;

    t += 1;
    for (int l = 0; l < L; l++) {
        M[l] = M[l] * beta_m + m_grad_J[l]                       * (1 - beta_m);
        V[l] = V[l] * beta_v + m_grad_J[l].hadamard(m_grad_J[l]) * (1 - beta_v);

        Matrix weight = m_layers[l].weights();
        for (size_t i = 0; i < weight.size(); i++) {
            for (size_t j = 0; j < weight[0].size(); j++) {
                double M_hat = M[l][i][j] / (1 - pow(beta_m, t));
                double V_hat = V[l][i][j] / (1 - pow(beta_v, t));
                weight[i][j] = weight[i][j] - (M_hat / (sqrt(V_hat) + 1e-8)) * _hyperparameters.learning_rate;
            }
        }
        m_layers[l].setWeights(weight);
    }

}

double MLP::CELoss(Matrix y_hot_one) {
    double loss = CELossFunction(m_layers.back().output(), y_hot_one).second;
    return loss;
}

double MLP::backwards(Matrix input, Matrix y_hot_one) {
    if (t == 0) {
        print("Epochs: Training Loss");
        print(t, ": ", CELoss(y_hot_one));
    }

    backpropagation(input, y_hot_one);
    Adam();

    //if (_hyperparameters.early_stopping) {
        double loss = CELoss(y_hot_one);
        if (t % 50 == 0)
            print(t, ": ", loss);
        return loss;
    /* }
    else {
        if (t % 50 == 0) {
            double loss = CELoss(y_hot_one);
            print(t, ": ", loss);
            return 
        }
        return 0.0;
    }*/
}



void MLP::saveWeights(const std::string& filename) {
    std::ofstream file(filename);
    for (auto& layer : m_layers) {
        Matrix W = layer.weights();
        for (auto& row : W) {
            for (double val : row)
                file << val << " ";
            file << "\n";
        }
        file << "===\n"; // Séparateur entre couches
    }
    file.close();
}

void MLP::loadWeights(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    int layer_index = 0;
    Matrix W;
    while (std::getline(file, line)) {
        if (line == "===") {
            m_layers[layer_index].setWeights(W);
            W.clear();
            layer_index++;
        }
        else {
            std::istringstream iss(line);
            dvector row;
            double val;
            while (iss >> val)
                row.push_back(val);
            W.push_back(row);
        }
    }
    file.close();
}
