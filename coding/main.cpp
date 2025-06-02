#include "MLP.h"
#pragma GCC diagnostic ignored "-Wnarrowing"


int main() {
    // Data init
    // Train
    dmatrix images;
    dvector labels;
    std::string trainImagesFile = "MNIST/train-images.idx3-ubyte";
    std::string trainLabelsFile = "MNIST/train-labels.idx1-ubyte";
    readMNIST(trainImagesFile, trainLabelsFile, images, labels);
    dmatrix labels_hotOnes = hotOne(labels, 10);
    // Test
    dmatrix imagesTests;
    dvector labelsTests;
    std::string testImagesFile = "MNIST/t10k-images.idx3-ubyte";
    std::string testLabelsFile = "MNIST/t10k-labels.idx1-ubyte";
    readMNIST(testImagesFile, testLabelsFile, imagesTests, labelsTests);
    print("Finished loading.");

    // MLP init
    dvector hidden_layers_sizes{ 728, 256, 128, 10 };
    MLP NN(hidden_layers_sizes);

    // Losses init
    dvector trainLosses;
    dvector testLosses;
    dvector accuracies;

    // Train:
    int epochs = 1.f / 64.f * images.size(); // So we have 60000/64 = 900 epochs
    print(epochs, " epochs.");
    for (int i = 0; i < epochs; i++) {
        dmatrix x_train(&images[64 * i], &images[64 * (i + 1)]);
        dmatrix y_test(&labels_hotOnes[64 * i], &labels_hotOnes[64 * (i + 1)]);

        // Training with batch samples
        dmatrix y_train = NN.forward(x_train);
        NN.backwards(x_train, y_test);

        // testing & accuracy each five epochs (because it would take too much time to compute test_loss at each epoch
        if (i % 5 == 0) {
            // Accuracy
            int acc = 0;
            for (int j = 0; j < 64; j++)
                if (y_test[j] == getCertitude(y_train)[j]) // Here, getCertitude takes the max for each vector(proba). Such as [0.4, 0.6]->[0, 1]
                    acc += 1;
            accuracies.push_back(100.f * acc / 64.f);

            // Train Loss
            trainLosses.push_back(NN.CELoss(y_test));

            // Test Loss!
            y_train = NN.forward(imagesTests);
            testLosses.push_back(NN.CELoss(hotOne(labelsTests, 10)));
        }
        else {
            accuracies.push_back(0);
            trainLosses.push_back(0);
            testLosses.push_back(0);
        }
        

    }

    writeFile(accuracies, trainLosses, testLosses, epochs, "training_data.csv");

}