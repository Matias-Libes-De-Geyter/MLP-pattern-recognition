#include "Matrix.h"

#define EULERS_NUMBER pow((1.0 + 1.0 / 10000000.0), 10000000.0)

struct hyperparameters {
	int input_dim;
	int output_dim;
	dvector hidden_layers_sizes;
	int epochs;
	int mini_batch_size;
	double learning_rate;
	double adam_beta_m;
	double adam_beta_v;
	double dropout_rate;
	bool early_stopping;
	int patience;
	bool learn;
	bool test;
	bool store_data;
};

double random(const double& min, const double& max);
Matrix hotOne(const dvector& y, const int& nElements); // Returns the "hot one" matrix of a vector.
std::pair<dvector, double> CELossFunction(const Matrix& y_pred, const Matrix& y_true); // Return the cross-entropy loss.

// Use parameter pack to easily print whatever the args are.
template<typename... Args> void print(const Args&... args) { (std::cout << ... << args) << std::endl; }
// Print function to print matrices. If type is not a matrix or a vector, it calls the other "print" function.
template<typename T>
void print(const T& container) {
	if constexpr (std::is_same_v<T, Matrix> || std::is_same_v<T, dvector>) {
		std::cout << "[";
		bool first = true;
		for (auto element : container) {
			if constexpr (!std::is_same_v<T, Matrix>) std::cout << (!first ? ", " : "") << element;
			else print(element);
			first = false;
		}
		std::cout << "]," << std::endl;
	}
	else print(container, "");
}
