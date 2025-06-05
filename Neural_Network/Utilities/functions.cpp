#include "functions.h"

double random(const double& min, const double& max) {
	std::mt19937_64 rng{};
	rng.seed(std::random_device{}());
	return std::uniform_real_distribution<>{min, max}(rng);
}
Matrix hotOne(const dvector& y, const int& nElements) {

	Matrix C(y.size(), nElements);

	for (int i = 0; i < y.size(); i++)
		C[i][y[i]] = 1;

	return C;
}
std::pair<dvector, double> CELossFunction(const Matrix& y_pred, const Matrix& y_true) {
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