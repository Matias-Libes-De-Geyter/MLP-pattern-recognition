#include "MLP.cuh"

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

    m_deltas[L - 1] = m_layers.back().output() - y_hot_one;
    m_grad_J[L - 1] = m_layers[L - 2].output().addBias_then_T() * m_deltas[L - 1];

    for (int l = L - 2; l >= 0; l--) {
        m_deltas[l] = (m_deltas[l + 1] * m_layers[l + 1].weights().T_then_removeBias()).hadamard(m_layers[l].preactivation().derivReLU());
        m_grad_J[l] = (l == 0 ? input : m_layers[l - 1].output()).addBias_then_T() * m_deltas[l];
    }

}

// ========================================================== ADAM OPTIMIZER IMPLEMENTATION
__global__ void adamKernel(double* W, double* M, double* V, const double* G, int N, hyperparameters hyp, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    M[idx] = hyp.adam_beta_m * M[idx] + (1.0 - hyp.adam_beta_m) * G[idx];
    V[idx] = hyp.adam_beta_v * V[idx] + (1.0 - hyp.adam_beta_v) * G[idx] * G[idx];

    double m_hat = M[idx] / (1 - pow(hyp.adam_beta_m, t));
    double v_hat = V[idx] / (1 - pow(hyp.adam_beta_v, t));

    W[idx] -= hyp.learning_rate * (m_hat / (sqrt(v_hat) + 1e-8));
}
void MLP::Adam() {
    t += 1;

    // Implementation of CUDA framework for backprop
    for (int l = 0; l < L; ++l) {
        Matrix& W_mat = m_layers[l].weights();
        Matrix& M_mat = M[l];
        Matrix& V_mat = V[l];
        Matrix& G_mat = m_grad_J[l];

        int rows = static_cast<int>(W_mat.size());
        int cols = static_cast<int>(W_mat[0].size());
        int N = rows * cols;
        size_t bytes = N * sizeof(double);

        std::vector<double> flat_W; flat_W.reserve(N);
        std::vector<double> flat_M; flat_M.reserve(N);
        std::vector<double> flat_V; flat_V.reserve(N);
        std::vector<double> flat_G; flat_G.reserve(N);

        for (auto& row : W_mat) flat_W.insert(flat_W.end(), row.begin(), row.end());
        for (auto& row : M_mat) flat_M.insert(flat_M.end(), row.begin(), row.end());
        for (auto& row : V_mat) flat_V.insert(flat_V.end(), row.begin(), row.end());
        for (auto& row : G_mat) flat_G.insert(flat_G.end(), row.begin(), row.end());

        double* d_W, * d_M, * d_V, * d_G;
        cudaMalloc(&d_W, bytes); cudaMalloc(&d_M, bytes);
        cudaMalloc(&d_V, bytes); cudaMalloc(&d_G, bytes);

        cudaMemcpy(d_W, flat_W.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_M, flat_M.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, flat_V.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_G, flat_G.data(), bytes, cudaMemcpyHostToDevice);

        int TPB = 256;
        int nBlocks = (N + TPB - 1) / TPB;
        adamKernel<<<nBlocks, TPB>>>(d_W, d_M, d_V, d_G, N, _hyperparameters, t);
        cudaDeviceSynchronize();

        std::vector<double> new_W(N), new_M(N), new_V(N);
        cudaMemcpy(new_W.data(), d_W, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(new_M.data(), d_M, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(new_V.data(), d_V, bytes, cudaMemcpyDeviceToHost);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int idx = i * cols + j;
                W_mat[i][j] = new_W[idx];
                M_mat[i][j] = new_M[idx];
                V_mat[i][j] = new_V[idx];
            }
        }
        m_layers[l].setWeights(W_mat);

        // Free memory
        cudaFree(d_W);
        cudaFree(d_M);
        cudaFree(d_V);
        cudaFree(d_G);
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

    double loss = CELoss(y_hot_one);
    if (t % 50 == 0)
        print(t, ": ", loss);
    return loss;
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
