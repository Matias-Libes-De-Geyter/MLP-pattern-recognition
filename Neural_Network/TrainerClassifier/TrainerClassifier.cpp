#include "TrainerClassifier.h"

// Load the MNIST database
int reverseInt(int i) { // To little-endian
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
void readMNIST(const std::string& imageFile, const std::string& labelFile,
	dmatrix& images, dvector& labels) {
	std::ifstream imgFile(imageFile, std::ios::binary);
	std::ifstream lblFile(labelFile, std::ios::binary);

	int magicNumber, numImages, numRows, numCols;
	imgFile.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
	magicNumber = reverseInt(magicNumber);
	imgFile.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
	numImages = reverseInt(numImages);
	imgFile.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
	numRows = reverseInt(numRows);
	imgFile.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));
	numCols = reverseInt(numCols);

	int labelMagicNumber, numLabels;
	lblFile.read(reinterpret_cast<char*>(&labelMagicNumber), sizeof(labelMagicNumber));
	labelMagicNumber = reverseInt(labelMagicNumber);
	lblFile.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
	numLabels = reverseInt(numLabels);

	int n = std::min(numImages, numLabels);
	images.resize(n, std::vector<double>(numRows * numCols));
	labels.resize(n);

	for (int i = 0; i < n; ++i) {
		// Images
		for (int j = 0; j < numRows * numCols; ++j) {
			unsigned char pixel;
			imgFile.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
			images[i][j] = static_cast<double>(pixel) / 255.0;
		}
		// Labels
		unsigned char label;
		lblFile.read(reinterpret_cast<char*>(&label), sizeof(label));
		labels[i] = static_cast<double>(label);
	}
}

void writeFile(const dvector& accuracies, const dvector& trainLosses, int nb_epochs, const std::string& filename) {
	std::ofstream outFile(filename);
	if (!outFile) {
		std::cerr << "Error opening file for writing: " << filename << std::endl;
		return;
	}

	outFile << "Epoch,Accuracy,TrainLoss\n";
	for (int epoch = 0; epoch < nb_epochs; ++epoch) {
		outFile << epoch + 1 << "," << accuracies[epoch] << "," << trainLosses[epoch] << "\n";
	}

	outFile.close();
}

void train(MLP& NN, const hyperparameters& hyper) {

    // Train
    dmatrix images;
    dvector labels;
    std::string trainImagesFile = "database/MNIST/train-images.idx3-ubyte";
    std::string trainLabelsFile = "database/MNIST/train-labels.idx1-ubyte";
    readMNIST(trainImagesFile, trainLabelsFile, images, labels);
	Matrix labels_hotOnes = hotOne(labels, 10);
    print("Database finished loading.");
	// Epochs
    int epochs = (hyper.epochs == 0 ? (60000.f/hyper.mini_batch_size) : hyper.epochs);
    print("Number of epochs: ", epochs); print("");

	// Store data init
	dvector trainLosses;
	dvector accuracies;
	
	// Early stopping
	MLP best(hyper);
	int iPatience = 0;
	double bestLoss = 2;
	
	// Train loop
    for (int i = 0; i < epochs; i++) {
        dmatrix x_train(&images[hyper.mini_batch_size * i], &images[hyper.mini_batch_size * (i + 1)]);
        dmatrix y_test(&labels_hotOnes[hyper.mini_batch_size * i], &labels_hotOnes[hyper.mini_batch_size * (i + 1)]);

        Matrix y_train = NN.forward(Matrix(x_train));
		double loss = NN.backwards(x_train, y_test);

		// Store loss and accuracy data in a .csv file for plotting
		if (hyper.store_data) {
			// Accuracy
			int acc = 0;
			for (int j = 0; j < hyper.mini_batch_size; j++) if (y_test[j] == y_train.setMaxToOne()[j]) acc += 1;
			accuracies.push_back(100.f * acc / hyper.mini_batch_size);

			// Train Loss
			trainLosses.push_back(loss);
		}

		// Implement early stopping
		if (hyper.early_stopping) {
			if (loss < bestLoss) {
				bestLoss = loss;
				best = NN;
				iPatience = 0;
				print("=== update");
			}
			else {
				iPatience++;
			}
			if (iPatience > hyper.patience) {
				NN = best;
				print("Break"); break;
			}
		}
    }

	if (hyper.store_data)
		writeFile(accuracies, trainLosses, epochs, "training_data.csv");
	

}

void test(MLP& NN, const hyperparameters& hyper) {

	// Tests
	dmatrix imagesTests;
	dvector labelsTests;
	std::string testImagesFile = "database/MNIST/t10k-images.idx3-ubyte";
	std::string testLabelsFile = "database/MNIST/t10k-labels.idx1-ubyte";
	readMNIST(testImagesFile, testLabelsFile, imagesTests, labelsTests);
	print("Test database finished loading.");

	dmatrix labelsTestsHot = hotOne(labelsTests, 10);
	dmatrix y_train_certain = NN.forward(imagesTests).setMaxToOne();

	print("");
	int acc = 0;
	for (int i = 0; i < y_train_certain.size(); i++)
		if (labelsTestsHot[i] == y_train_certain[i])
			acc += 1;
	print("The final loss is ", NN.CELoss(labelsTestsHot), ", with ", 100.f * acc / y_train_certain.size(), " % of accuracy");

}