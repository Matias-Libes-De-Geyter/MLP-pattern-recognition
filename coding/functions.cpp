#include "functions.h"

double random(const double& min, const double& max) {
	std::mt19937_64 rng{};
	rng.seed(std::random_device{}());
	return std::uniform_real_distribution<>{min, max}(rng);
}

void print(const dmatrix& arr, const std::string& texte) {
	std::cout << texte << std::endl << "[";
	for (const auto& row : arr) {
		std::cout << "[";
		for (const auto& value : row) {
			std::cout << value << ", ";
		}
		std::cout << "]," << std::endl;
	}
	std::cout << "]" << std::endl;
}
void print(const dmatrix& arr) {
	std::cout << "[";
	for (const auto& row : arr) {
		std::cout << "[";
		for (const auto& value : row) {
			std::cout << value << ", ";
		}
		std::cout << "]," << std::endl;
	}
	std::cout << "]" << std::endl;
}
void print(const dvector& arr) {
	std::cout << "[";
	for (const auto& value : arr) {
		std::cout << value << ", ";
	}
	std::cout << "]" << std::endl;
}
void print(const double& val) {
	std::cout << val << std::endl;
}
void print(const std::string& texte) {
	std::cout << texte << std::endl;
}
void print(const double& val, const std::string& texte) {
	std::cout << val << " " << texte << std::endl;
}
void printSize(const dmatrix& mat, int l, std::string type) {
	std::cout << type << " " << l << " is " << mat.size() << "x" << mat[0].size() << std::endl;
}
dmatrix operator*(const dmatrix& A, const dmatrix& B) {
	size_t n_rows = A.size();
	size_t n_columns = B[0].size();
	size_t n_inner = B.size();

	dmatrix C(n_rows, dvector(n_columns, 0));
	for (size_t i = 0; i < n_rows; i++) {
		for (size_t j = 0; j < n_columns; j++) {
			for (size_t k = 0; k < n_inner; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}

	return C;
}
dmatrix operator*(const double& a, const dmatrix& B) {
	size_t n_rows = B.size();
	size_t n_columns = B[0].size();

	dmatrix C(n_rows, dvector(n_columns, 0));
	for (size_t i = 0; i < n_rows; i++) {
		for (size_t j = 0; j < n_columns; j++) {
			C[i][j] = a * B[i][j];
		}
	}

	return C;
}
dmatrix operator+(const dmatrix& A, const dmatrix& B) {
	size_t n_rows = A.size();
	size_t n_columns = A[0].size();

	dmatrix C(n_rows, dvector(n_columns, 0));
	for (size_t i = 0; i < n_rows; i++) {
		for (size_t j = 0; j < n_columns; j++) {
			C[i][j] += A[i][j] + B[i][j];
		}
	}

	return C;
}
dmatrix operator+(const dmatrix& A, const dvector& B) {
	size_t n_rows = A.size();
	size_t n_columns = A[0].size();

	dmatrix C(n_rows, dvector(n_columns, 0));
	for (size_t i = 0; i < n_rows; i++) {
		for (size_t j = 0; j < n_columns; j++) {
			C[i][j] += A[i][j] + B[j];
		}
	}

	return C;
}
dvector operator+(const dvector& A, const dvector& B) {
	size_t n_rows = A.size();

	dvector C(n_rows, 0);
	for (size_t i = 0; i < n_rows; i++) {
		C[i] += A[i] + B[i];
	}

	return C;
}
dmatrix operator-(const dmatrix& A, const dmatrix& B) {
	size_t n_rows = A.size();
	size_t n_columns = A[0].size();

	dmatrix C(n_rows, dvector(n_columns, 0));
	for (size_t i = 0; i < n_rows; i++) {
		for (size_t j = 0; j < n_columns; j++) {
			C[i][j] += A[i][j] - B[i][j];
		}
	}

	return C;
}
dmatrix transpose(const dmatrix& A) {
	size_t n_rows = A.size();
	size_t n_columns = A[0].size();

	dmatrix C(n_columns, dvector(n_rows, 0));
	for (size_t i = 0; i < n_columns; i++) {
		for (size_t j = 0; j < n_rows; j++) {
			C[i][j] = A[j][i];
		}
	}

	return C;
}
dmatrix hadamard(const dmatrix& A, const dmatrix& B) {
	size_t n_rows = A.size();
	size_t n_columns = A[0].size();

	dmatrix C(n_rows, dvector(n_columns, 0));
	for (size_t i = 0; i < n_rows; i++) {
		for (size_t j = 0; j < n_columns; j++) {
			C[i][j] += A[i][j] * B[i][j];
		}
	}

	return C;
}


dmatrix aug_inputs_mult(const dmatrix& A, const dmatrix& B) {
	size_t n_rows = A.size();
	size_t n_columns = B[0].size();
	size_t n_inner = B.size();

	dmatrix C(n_rows, dvector(n_columns, 0));
	for (size_t i = 0; i < n_rows; i++) {
		for (size_t j = 0; j < n_columns; j++) {
			for (size_t k = 0; k < n_inner - 1; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
			C[i][j] += 1 * B[n_inner - 1][j]; // Comme si on avait rajouté une colonne à "input" de que des 1.
		}
	}

	return C;
}
dmatrix addBiases(const dmatrix& A) {
	size_t n_rows = A.size();
	size_t n_columns = A[0].size();

	dmatrix C(n_rows, dvector(n_columns + 1, 0));
	for (size_t i = 0; i < n_rows; i++) {
		for (size_t j = 0; j < n_columns; j++) {
			C[i][j] = A[i][j];
		}
		C[i][n_columns] = 1;
	}

	return C;
}
dmatrix remove_bias(const dmatrix& A) {
	size_t n_rows = A.size();
	size_t n_columns = A[0].size();

	dmatrix C(n_rows, dvector(n_columns - 1, 0));
	for (size_t i = 0; i < n_rows; i++) {
		for (size_t j = 0; j < n_columns - 1; j++) {
			C[i][j] = A[i][j];
		}
	}

	return C;
}
dmatrix ReLU_derivate(const dmatrix& A) {
	size_t n_rows = A.size();
	size_t n_columns = A[0].size();

	dmatrix C(n_rows, dvector(n_columns, 0));
	for (size_t i = 0; i < n_rows; i++) {
		for (size_t j = 0; j < n_columns; j++) {
			C[i][j] = (A[i][j] > 0.0 ? 1.0 : 0);
		}
	}
	return C;
}

std::pair<dvector, double> CELossFunction(const dmatrix& y_pred, const dmatrix& y_true) {
	dvector loss(y_pred.size(), 1e-9);
	double mean_loss = 0.0;

	for (int i = 0; i < y_pred.size(); i++)
		for (int j = 0; j < y_pred[0].size(); j++)
			if (y_true[i][j] == 1.0) {
				double prob = std::max(y_pred[i][j], 1e-9);
				loss[i] = -log(prob);
				mean_loss += loss[i];
				break;
			}

	mean_loss /= y_pred.size();  // moyenne sur tous les exemples

	return { loss, mean_loss };
}

dmatrix getCertitudeHot(const dmatrix& A) {
	size_t n_rows = A.size();
	size_t n_columns = A[0].size();

	dmatrix C(n_rows, dvector(n_columns, 0));
	for (size_t i = 0; i < n_rows; i++) {

		auto maxElement = std::max_element(A[i].begin(), A[i].end());
		size_t maxIndex = std::distance(A[i].begin(), maxElement);

		C[i][maxIndex] = 1;
	}

	return C;
}
dmatrix getCertitude(const dmatrix& A) {
	size_t n_rows = A.size();
	size_t n_columns = A[0].size();

	dmatrix C(n_rows, dvector(n_columns, 0));
	for (size_t i = 0; i < n_rows; i++) {

		auto maxElement = std::max_element(A[i].begin(), A[i].end());
		size_t maxIndex = std::distance(A[i].begin(), maxElement);

		C[i][maxIndex] = 1;
	}

	return C;
}


dmatrix hotOne(const dvector& y, const int& nElements) {

	dmatrix C(y.size(), dvector(nElements, 0));

	for (int i = 0; i < y.size(); i++) {
		C[i][y[i]] = 1;
	}
	return C;

}


int reverseInt(int i) {
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

	// Get sure that sizes matches
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
void writeFile(const dvector& accuracies, const dvector& trainLosses, const dvector& testLosses, int nb_epochs, const std::string& filename) {
	std::ofstream outFile(filename);
	if (!outFile) {
		std::cerr << "Error opening file for writing: " << filename << std::endl;
		return;
	}

	outFile << "Epoch,Accuracy,TrainLoss,TestLoss\n";
	for (int epoch = 0; epoch < nb_epochs; ++epoch) {
		outFile << epoch + 1 << "," << accuracies[epoch] << "," << trainLosses[epoch] << "," << testLosses[epoch] << "\n";
	}

	outFile.close();
}