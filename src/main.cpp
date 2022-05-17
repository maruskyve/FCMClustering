/*
 * Created by MarufN on 24/04/2022.
 */

#include <bits/stdc++.h>
#include "data.h"

using namespace std;

typedef vector<vector<float>> FMatrix;
typedef vector<float> FVec;

class FCMClustering {
private:
	FMatrix x, u, v;
	FVec p;

	// H-Param
	int cVal, maxIter, power;
	float minError;

	int x_nRow, x_nCol;
	bool randomizeU = false;

public:
	FCMClustering(FMatrix x,
	              int c_val,
	              int power,
	              float min_error,
	              int max_iter,
	              FMatrix u = {}) {
		this->x = move(x);
		this->cVal = c_val;
		this->maxIter = max_iter;
		this->power = power;
		this->minError = min_error;
		this->x_nRow = this->x.size();
		this->x_nCol = this->x[0].size();

		if (u.empty()) {
			this->u.resize(x_nRow, FVec(cVal));
			randomizeU = true;
		} else
			this->u = move(u);
		v.resize(cVal, FVec(x_nCol));
		p.resize(maxIter);

		try {
			modelBuilder();
		} catch (exception &e) {
			cerr << e.what();
		}
	}

	~FCMClustering();

private:
	static FMatrix transpose(FMatrix array) {
		FMatrix outputArray(array[0].size(), FVec(array.size()));

		for (int i = 0; i < outputArray.size(); i++) {
			for (int j = 0; j < outputArray[i].size(); j++) {
				outputArray[j][i] = array[i][j];
			}
		}
		return outputArray;
	}

	void URandomization() {  // Flagged as OK
		/*
		 * Method (private) untuk U-Matrix initialization
		 */

		printf("Initializing U-Matrix random values...\n");
		int maxRecursiveIter = 5;
		float fMin = .0, fMax = (float) 1 / (float) cVal;

		unsigned seed = chrono::system_clock::now().time_since_epoch().count();
		default_random_engine gen(seed);
		uniform_real_distribution<float> dist(fMin, fMax);

		for (int i = 0; i < x_nRow; ++i) {
			float flagSum = .0;

			inf_handler:  // Infinite recursive solver handler
			int recCount = 0;

			for (int j = 0; j < cVal; ++j) {
				u[i][j] = dist(gen);
				flagSum += u[i][j];
			}
			recursive_solver:
			if (flagSum > 1.0) {
				float maxVal = *max_element(u[i].begin(), u[i].end());
				int argMax = max_element(u[i].begin(), u[i].end()) - u[i].begin();
				u[i][argMax] = abs(1.0 - (flagSum - maxVal));
//				cout << "ENTER MAX SOLVER (" << i << ")";
			} else if (flagSum < 1.0) {
				float minVal = *min_element(u[i].begin(), u[i].end());
				int argMin = min_element(u[i].begin(), u[i].end()) - u[i].begin();
				u[i][argMin] = abs((1.0 - flagSum) + minVal);
//				cout << "ENTER MIN SOLVER (" << i << ")";
			}
			flagSum = .0, recCount++;
			for (int k = 0; k < cVal; ++k) {
//				cout << u[i][k] << " + ";
				flagSum += u[i][k];
			}
//			cout << endl;
			if (flagSum != 1.0) {
				if (recCount >= maxRecursiveIter)
					goto inf_handler;
				goto recursive_solver;
			}
//			cout << flagSum << endl;
		}
	}

	void computeV() {
		/*
		 * Method (private) untuk menghitung titik pusat masing-masing cluster terhadap n-atribut
		 */

		for (int i = 0; i < cVal; ++i) {  // equal to n_cluster
			for (int j = 0; j < x_nCol; ++j) {  // equal to n_xAttrib
				float T = .0, B = .0;

				// Sigma(T,B)
				for (int k = 0; k < x_nRow; ++k) {
					T += (float) pow(u[k][i], power) * x[k][j];
					B += pow(u[k][i], power);
				}
				v[i][j] = T / B;
//				cout << v[i][j] << ", ";
			}
		}
	}

	void computeP(int t) {
		/*
		 * Method (private) untuk menghitung Fungsi Objektif / Pt
		 */

		for (int i = 0; i < x_nRow; ++i) {  // accessing row of u
			for (int j = 0; j < cVal; ++j) { // accessing col of u
				float xvSum = .0;

				// Summation of (Xij - Vij)^2
				for (int k = 0; k < x_nCol; ++k) {
					xvSum += (float) pow(x[i][k] - v[j][k], 2);
				} // cout << endl;
				xvSum *= pow(u[i][j], power);
				p[t] += xvSum;
//				cout << xvSum << ", ";
			}
//			cout << endl;
		}

	}

	void updateU() {  // Flagged as OK
		/*
		 * Method (private) untuk memperbaiki matriks U
		 */
		for (int i = 0; i < x_nRow; ++i) {
			FVec L(cVal);
			float LT = .0;

			for (int j = 0; j < cVal; ++j) {  // Find cumulative of T
				for (int k = 0; k < x_nCol; ++k)
					L[j] += pow(x[i][k] - v[j][k], 2);

				L[j] = pow(L[j], -1 / (power - 1));
				LT += L[j];
			}
			for (int l = 0; l < cVal; ++l)  // Flagged as OK
				u[i][l] = L[l] / LT;
		}
	}

	void modelBuilder() {
		/*
		 * FCM Clustering Algorithm
		 */
		if (randomizeU)
			URandomization();
//		printf("---------- Compute V START ----------\n");
//		computeV();
//		printf("---------- Compute V END ----------\n---------- Compute P START ----------\n");
//		computeP(0);
//		printf("---------- Compute P END ----------\n---------- Update U START ----------\n");
//		updateU();
//		printf("---------- Compute U END ----------\n");
		for (int i = 0; i < maxIter; ++i) {
			printf("\n-------------- Iteration %i --------------\n", i + 1);
			computeV();
			computeP(i);
			updateU();

			float c = i > 0 ? abs(p[i] - p[i - 1]) : p[i] - 0;
			printf("p[%i]: %f\nminError: %f\nerror: %f\n", i, p[i], minError, c);

			if ((i > 0 ? abs(p[i] - p[i - 1]) : p[i] - 0) < minError) {
				cout << "\n### Stopping criterion reached, iteration stopped ###\n\n";
				break;
			} else continue;
		}
	}

public:
	FMatrix getClustersCenter() {
		return v;
	}

	FMatrix getFuzzyClusters() {
		return u;
	}

	void printClustersCenter() {
		cout << "Seq -> Clusters, Attributes Center Clusters = ,\n";
		for (int i = 0; i < cVal; ++i) {
			string num = to_string(i + 1);

			for (int k = num.length(); k < to_string(x_nCol).length(); ++k)
				cout << " ";
			cout << num << " ";

			for (int j = 0; j < x_nCol; ++j) {
				printf("| %f %s", v[i][j], (j == x_nCol - 1 ? "|" : ""));
			}
			cout << endl;
		}
	}

	void printFuzzyClusters(bool mode_binary = false) {
		cout << "Seq -> Num, Clusters = ";
		for (int i = 0; i < cVal; ++i)
			cout << i + 1 << (i == cVal - 1 ? "" : ", ")
			     << (i == cVal - 1 ? "\n" : "");

		for (int l = 0; l < x_nRow; ++l) {
			float mVal = *max_element(u[l].begin(), u[l].end());
			string num = to_string(l + 1);

			for (int i = num.length(); i < to_string(x_nRow).length(); ++i)
				num += " ";
			cout << num << " ";

			for (int j = 0; j < cVal; ++j) {
				cout << "| " << (mode_binary ? (u[l][j] == mVal ? "1" : " ") : to_string(u[l][j]))
				     << (j < cVal - 1 ? " " : " |");
			}
			cout << endl;

		}
	}
};

FCMClustering::~FCMClustering() = default;

int main() {
	FMatrix x = {{15000.000, 25000.000, 42, 5000.000},
	             {20000.000, 26420.000, 72, 5320.000},
	             {17820.000, 22052.000, 35, 5200.000},
	             {16205.000, 18500.000, 12, 4250.000},
	             {8000.000,  15200.000, 5,  3500.000},
	             {14260.000, 19640.000, 15, 4023.000}};

	FCMClustering m1(data1_x(), 5, 2, pow(10, -5), 100, data1_u());
//	FCMClustering m1(dataBreastCancer_x, 2, 2, .000001, 500);
//	FCMClustering m1(dataPublicClustering_x(), 30, 2, .000000001, 2000);
//	FCMClustering m1(dataWaterPotability(), 2, 2, .0000001, 1000);
//	FCMClustering m1(dataBQuery_c9a_x(), 10, 2, .0000000000001, 6000);
//	FCMClustering m1(dataBQuery_c7a_x(), 10, 2, .0000000000001, 5000);

//	FMatrix clustersCenter = m1.getClustersCenter();
	m1.printClustersCenter();
	m1.printFuzzyClusters(true);
	system("pause");
	return 0;
}
