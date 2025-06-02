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
    std::cout << images.size() << " images, et" << std::endl;
    std::cout << labels[0] << " labels." << std::endl;
    print("Finished loading");


    dvector hidden_layers_sizes{ 728, 256, 128, 10 };
    MLP NN(hidden_layers_sizes);

    // prendre des batchs de 64 échantillons.
    // epochs should be 20

    // Train:
    int epochs = 1.f / 64.f * images.size();
    print(epochs);
    dmatrix labels_hotOnes = hotOne(labels, 10);
    for (int i = 0; i < epochs; i++) {
        dmatrix::const_iterator first_img = images.begin() + 64 * i;
        dmatrix::const_iterator last_img = images.begin() + 64 * (i + 1);
        dmatrix x_train(first_img, last_img);

        dmatrix::const_iterator first_label = labels_hotOnes.begin() + 64 * i;
        dmatrix::const_iterator last_label = labels_hotOnes.begin() + 64 * (i + 1);
        dmatrix y_test(first_label, last_label);

        dmatrix y_train = NN.forward(x_train);
        NN.backwards(x_train, y_test);

        //ALT: std::vector<int>   sub(&data[100000],&data[101000]);
    }

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