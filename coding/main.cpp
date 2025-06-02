#include "MLP.h"
#pragma GCC diagnostic ignored "-Wnarrowing"


int main()
{
    dmatrix images;
    dvector labels;

    // Get MNIST images and labels
    std::string trainImagesFile = "MNIST/train-images.idx3-ubyte";
    std::string trainLabelsFile = "MNIST/train-labels.idx1-ubyte";
    readMNIST(trainImagesFile, trainLabelsFile, images, labels);
    std::cout << images.size() << " images, and " << labels[0] << " labels." << std::endl;
    print("Finished loading.");


    dvector hidden_layers_sizes{ 728, 256, 128, 10 };
    MLP NN(hidden_layers_sizes);

    // prendre des batchs de 64 échantillons.
    // epochs should be 20

    dvector accuracies;
    dvector losses;

    // Train:
    int epochs = 1.f / 64.f * images.size(); // divide epochs by batches of 64.
    print(epochs, " epochs.");
    dmatrix labels_hotOnes = hotOne(labels, 10);
    for (int i = 0; i < epochs; i++) {
        dmatrix x_train(&images[64 * i], &images[64 * (i + 1)]);
        dmatrix y_test(&labels_hotOnes[64 * i], &labels_hotOnes[64 * (i + 1)]);

        dmatrix y_train = NN.forward(x_train);
        losses.push_back(NN.backwards(x_train, y_test));

        // Accuracy:
        dmatrix y_train_certain = getCertitude(y_train);
        int acc = 0;
        for (int j = 0; j < 64; j++)
            if (y_test[j] == y_train_certain[j])
                acc += 1;
        
        accuracies.push_back(100.f * acc / 64.f);

    }

    writeFile(accuracies, losses, epochs, "training_data.csv");

    // Test:
    dmatrix imagesTests;
    dvector labelsTests;
    std::string testImagesFile = "MNIST/t10k-images.idx3-ubyte";
    std::string testLabelsFile = "MNIST/t10k-labels.idx1-ubyte";

    readMNIST(testImagesFile, testLabelsFile, imagesTests, labelsTests);
    dmatrix labelsTestsHot = hotOne(labelsTests, 10);
    dmatrix y_train_certain = getCertitude(NN.forward(imagesTests));


    int acc = 0;
    for (int i = 0; i < y_train_certain.size(); i++) {
        if (i % 1000 == 0) print(i);
        if (labelsTestsHot[i] == y_train_certain[i]) {
            acc += 1;
        }
    }
    print(100.f * acc / y_train_certain.size(), "% d'accuracy");

}