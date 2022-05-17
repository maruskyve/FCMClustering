#include <bits/stdc++.h>
#include "data.h"

using namespace std;

typedef vector<vector<float>> FMatrix;
typedef vector<float> FVec;

class FCMClustering {
private:
	FMatrix x, u, v;
	FVec p;
	int cVal, maxIter, power;
	float minError;
	int x_nRow, x_nCol;

public:
	FCMClustering(FMatrix x, int c_val, int power, float min_error, FMatrix tmp_u, int max_iter) {
		this->x = std::move(x);
		this->cVal = c_val;
		this->maxIter = max_iter;
		this->power = power;
		this->minError = min_error;
		this->x_nRow = this->x.size();
		this->x_nCol = this->x[0].size();

		this->u = std::move(tmp_u);
//		u.resize(x_nRow, FVec(cVal));
		v.resize(cVal, FVec(x_nCol));
		p.resize(maxIter);

		try {
			buildModel();
		} catch (exception &e) {
			cerr << e.what();
		}
	}

private:
	void URandomization() {
//		srand(time(NULL));

		for (int i = 0; i < x_nRow; ++i) {
			for (int j = 0; j < cVal; ++j) {
				u[i][j] = (float) rand() / (RAND_MAX);
			}
		}
	}

	void computeV() {
		/*
		 * Method (private) untuk menghitung titik pusat masing-masing cluster terhadap n-atribut
		 */
//		for (int m = 0; m < x_nRow; ++m) {
//			for (int i = 0; i < cVal; ++i) {
//				cout << u[m][i] << ", ";
//
//			} cout << endl;
//
//		} cout << "uMat\n";

		for (int i = 0; i < cVal; ++i) {  // equal to n_cluster
			for (int j = 0; j < x_nCol; ++j) {  // equal to n_xAttrib
				float tmpValT = .0, tmpValB = .0;

				// Sigma(T,B)
				for (int k = 0; k < x_nRow; ++k) {
					for (int l = j; l < j + 1; ++l) {  // Iterating x with only col that same with j
						tmpValT += (float) pow(u[k][i], power) * x[k][l];
						tmpValB += pow(u[k][i], power);
					}
				}
				v[i][j] = tmpValT / tmpValB;
				cout << v[i][j] << ", ";
			}
			cout << endl;
		}
	}

	void computeP(int t) {
		/*
		 * Method untuk menghitung Fungsi Objektif
		 */
		float lSum = .0;

		for (int i = 0; i < x_nRow; ++i) {  // accessing row of u
			for (int j = 0; j < cVal; ++j) { // accessing col of u

				// Summation of (Xij - Vij)^2
				for (int k = 0; k < x_nCol; ++k) {
					lSum += (float) (pow(x[i][k] - v[j][k], 2) * pow(u[i][j], power));
				}
			}
		}
		p[t] = lSum;
//		p.push_back(lSum);
		cout << p[t] << endl;
	}

	void updateU() {  // MARKED AS PROBLEM
		FMatrix lMat(x_nRow, FVec(cVal));
		FVec sum_(x_nRow);

		for (int i = 0; i < x_nRow; ++i) {
			for (int j = 0; j < cVal; ++j) {
				float T, B = .0;

//				for (int k = 0; k < x_nCol; ++k) {
//					lMat[i][j] += (float) (pow(x[i][k] - v[j][k], 2));
//					lMat[i][j] = pow(lMat[i][j], -1);
//				}
//				sum_[i] += lMat[i][j];
//				cout << lMat[i][j] << ", ";

				for (int k = 0; k < x_nCol; ++k) {
					T += (float) (pow(x[i][k] - v[j][k], 2));
					T = pow(T, -1 / (power - 1));
//					printf("pow(%f - %f)^2 = %f\n", x[i][k], v[j][k], T);
				}
//				cout <<endl;

				u[i][j] = T;
				for (int l = 0; l < cVal; ++l) {
					for (int k = 0; k < x_nCol; ++k) {
						B += pow(x[i][k] - v[j][k], 2);
					}
					B = pow(B, -1 / (power - 1));
				}
//				u[i][j] = T / B;

				// Temp
//				u[i][j] = stof("0." + to_string((int) (u[i][j] * 1000 + 0.5)));
				cout << u[i][j] << ", ";
				// Temps
			}
			cout << endl;
//			cout << sum_[i] << endl;
		}
//		for (int l = 0; l < x_nRow; ++l) {
//			for (int i = 0; i < cVal; ++i) {
//				u[l][i] = u[l][i]/sum_[i];
//				cout << u[l][i] << ", ";
//			} cout << endl;
//		}
	}

	void buildModel() {
//		URandomization();
		computeV();
		computeP(0);
		updateU();
//		for (int i = 0; i < maxIter; ++i) {
//			printf("\n-------------- Iteration %i --------------\n", i + 1);
//			computeV();
//			computeP(i);
//			updateU();
//
//			float c = i > 0 ? abs(p[i] - p[i - 1]) : p[i] - 0;
//			printf("minError: %f\npStop: %f\n", minError, c);
//
//			if ((i > 0 ? abs(p[i] - p[i - 1]) : p[i] - 0) < minError) {
//				cout << "Iter Stopped\n";
//				break;
//			} else continue;
//		}
	}

};

int main() {
	std::cout << "Hello, World!" << std::endl;
	FMatrix x = {{15000.000, 25000.000, 42, 5000.000},
	             {20000.000, 26420.000, 72, 5320.000},
	             {17820.000, 22052.000, 35, 5200.000},
	             {16205.000, 18500.000, 12, 4250.000},
	             {8000.000,  15200.000, 5,  3500.000},
	             {14260.000, 19640.000, 15, 4023.000}};

//	FCMClustering m1(data1_x, 5, 2, pow(10, -5), data1_u, 100);
	FCMClustering m1(data3_x, 3, 2, 0.1, data3_u, 100);
	return 0;
}
