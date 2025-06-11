#include "TrainerClassifier.cuh"
#pragma GCC diagnostic ignored "-Wnarrowing"


hyperparameters current_hyperparameters = {
    28 * 28,                  // input_dim
    10,                       // output_dim
    {256, 128},               // hidden_layers_sizes
    0,                        // epochs (0 = all the database)
    32,                       // mini_batch_size
    0.001,                    // learning_rate
    0.9,                      // adam_beta_m
    0.999,                    // adam_beta_v
    0.001,                    // dropout_rate
    false,                    // early_stopping
    150,                      // patience
    false,                    // learn
    true,                     // test
    false                      // store_data
};

int main() {
    // Ask for model training
    print("Train ? (y/n)"); char a; std::cin >> a;
    if (a == 'y') current_hyperparameters.learn = true;

    // MLP init
    MLP NN(current_hyperparameters);

    // Training
    if (current_hyperparameters.learn) {
        train(NN, current_hyperparameters);
        NN.saveWeights("model_weights.txt");
    }
    else {
        NN.loadWeights("model_weights.txt");
        print("Weights loaded !");
    }

    if (current_hyperparameters.test) test(NN, current_hyperparameters);

    return 0;
}