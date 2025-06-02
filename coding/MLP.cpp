#include "MLP.h"

MLP::MLP(const dvector& hidden_layers_sizes) : L(hidden_layers_sizes.size() - 1) {

    std::vector<dmatrix> matrix_L_sized(L);
    m_deltas = matrix_L_sized;
    m_grad_J = matrix_L_sized;

    for (int l = 0; l < L; l++) {
        DenseBlock layer(hidden_layers_sizes[l], hidden_layers_sizes[l + 1]);
        m_layers.push_back(layer);

        M.push_back(dmatrix(hidden_layers_sizes[l] + 1, dvector(hidden_layers_sizes[l + 1], 0.0)));
        V.push_back(M[l]);
    }
    t = 0;

};

dmatrix MLP::forward(const dmatrix& input) {

    m_layers[0].forward(input);
    for (int l = 1; l < L - 1; l++)
        m_layers[l].forward(m_layers[l - 1].output());
    m_layers[L - 1].forward(m_layers[L - 2].output(), "softmax");

    return m_layers.back().output();

}

void MLP::backpropagation(const dmatrix& input, const dmatrix& y_hot_one) {

    m_deltas[L - 1] = m_layers.back().output() - y_hot_one; // 100% sûr que les deux sont de dimension 15x3
    m_grad_J[L - 1] = transpose(addBiases(m_layers[L - 2].output())) * m_deltas[L - 1];


    for (int l = L - 2; l >= 0; l--) {
        m_deltas[l] = hadamard(m_deltas[l + 1] * remove_bias(transpose(m_layers[l + 1].weights())), ReLU_derivate(m_layers[l].preactivation()));
        m_grad_J[l] = transpose(addBiases(l == 0 ? input : m_layers[l - 1].output())) * m_deltas[l];
                                //important addBiases

        /*printSize(m_deltas[l], l, "Delta");
        printSize(m_grad_J[l], l, "Grad");
        printSize(m_layers[l].weights(), l, "Weights");
        std::cout << std::endl;
        std::cout << m_grad_J[l].size() - m_layers[l].weights().size() << std::endl;
        std::cout << m_grad_J[l][0].size() - m_layers[l].weights()[0].size() << std::endl;*/
    }

}

void MLP::Adam() {
    double lr = 0.1;

    double beta_m = 0.9;
    double beta_v = 0.999;

    t += 1;
    for (int l = 0; l < L; l++) {
        M[l] = beta_m * M[l] + (1 - beta_m) * m_grad_J[l];
        V[l] = beta_v * V[l] + (1 - beta_v) * hadamard(m_grad_J[l], m_grad_J[l]);

        dmatrix weight = m_layers[l].weights();
        for (size_t i = 0; i < weight.size(); i++) {
            for (size_t j = 0; j < weight[0].size(); j++) {
                double M_hat = M[l][i][j] / (1 - pow(beta_m, t));
                double V_hat = V[l][i][j] / (1 - pow(beta_v, t));
                weight[i][j] = weight[i][j] - lr * (M_hat / (sqrt(V_hat) + 1e-8));
            }
        }
        m_layers[l].setWeights(weight);
    }

}

void MLP::backwards(const dmatrix& input, const dmatrix& y_hot_one) {
    if(t == 0)
        print(CELoss(m_layers.back().output(), y_hot_one).second);
    
    backpropagation(input, y_hot_one);
    Adam();
    
    if(t % 100 == 0)
        print(CELoss(m_layers.back().output(), y_hot_one).second);
}