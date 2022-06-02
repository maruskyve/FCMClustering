/*
 * Created by MarufN on 24/04/2022.
 * C++14
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

typedef vector<vector<float>> F_Matrix;
typedef vector<float> F_Vec;


/**
 *  Modest Fuzzy C-Means (FCM) Clustering class
 */
class FCMClustering {
private:
	F_Matrix x, u, v;
	F_Vec p;

	// H-Param
	int c_val, max_iter, power;
	float min_error;

	int x_nRow{}, x_nCol{};
	bool randomize_u = false;

public:
	FCMClustering(int c_val,
	              int power,
	              float min_error,
	              int max_iter,
	              F_Matrix u = {}) {
		this->c_val = c_val;
		this->max_iter = max_iter;
		this->power = power;
		this->min_error = min_error;

		if (u.empty()) randomize_u = true;
		else this->u = std::move(u);
	}

	~FCMClustering();

private:
	void URandomization() {
		/*
		 * Method (private) untuk U-Matrix initialization
		 */

		printf("Initializing U-Matrix random values...\n");
		int max_re_iter = 5;  // Max recursive iter
		float f_min = .0, f_max = (float) 1 / (float) c_val;

		unsigned seed = chrono::system_clock::now().time_since_epoch().count();
		default_random_engine gen(seed);
		uniform_real_distribution<float> dist(f_min, f_max);

		for (int i = 0; i < x_nRow; ++i) {
			float flag_sum = .0;  // Flag summation

			inf_handler:  // Infinite recursive solver handler
			int re_ct = 0;  // Recursive count

			for (int j = 0; j < c_val; ++j) {
				u[i][j] = dist(gen);
				flag_sum += u[i][j];
			}
			recursive_solver:
			if (flag_sum > 1.0) {
				float maxVal = *max_element(u[i].begin(), u[i].end());
				int argMax = max_element(u[i].begin(), u[i].end()) - u[i].begin();
				u[i][argMax] = abs(1.0 - (flag_sum - maxVal));
//				cout << "ENTER MAX SOLVER (" << i << ")";
			} else if (flag_sum < 1.0) {
				float minVal = *min_element(u[i].begin(), u[i].end());
				int argMin = min_element(u[i].begin(), u[i].end()) - u[i].begin();
				u[i][argMin] = abs((1.0 - flag_sum) + minVal);
//				cout << "ENTER MIN SOLVER (" << i << ")";
			}
			flag_sum = .0, re_ct++;
			for (int k = 0; k < c_val; ++k) {
//				cout << u[i][k] << " + ";
				flag_sum += u[i][k];
			}
//			cout << endl;
			if (flag_sum != 1.0) {
				if (re_ct >= max_re_iter)
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

		for (int i = 0; i < c_val; ++i) {  // equal to n_cluster
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
//			cout << endl;
		}
	}

	void computeP(int t) {
		/*
		 * Method (private) untuk menghitung Fungsi Objektif / Pt
		 */

		for (int i = 0; i < x_nRow; ++i) {  // accessing row of u
			for (int j = 0; j < c_val; ++j) { // accessing col of u
				float xvSum = .0;

				// Summation of (Xij - Vij)^2
				for (int k = 0; k < x_nCol; ++k)  // Search summation of (Xij - Vij)^2
					xvSum += (float) pow(x[i][k] - v[j][k], 2);

				xvSum *= pow(u[i][j], power);
				p[t] += xvSum;
//				cout << xvSum << ", ";
			}
//			cout << endl;
		}

	}

	void updateU() {
		/*
		 * Method (private) untuk memperbaiki matriks U
		 */
		for (int i = 0; i < x_nRow; ++i) {
			F_Vec L(c_val);
			float LT = .0;

			for (int j = 0; j < c_val; ++j) {  // Compute summation of T
				for (int k = 0; k < x_nCol; ++k)
					L[j] += pow(x[i][k] - v[j][k], 2);

				L[j] = pow(L[j], -1 / (power - 1));
				LT += L[j];
			}
			for (int l = 0; l < c_val; ++l)  // Compute the AVG for each L
				u[i][l] = L[l] / LT;
		}
	}

	/**
	 * FCM One-Hot Array Generator
	 * @param lookup_array
	 * @return
	 */
	F_Matrix FCMOneHot(F_Matrix &lookup_array) const {
		F_Matrix one_hot_c(x_nRow, F_Vec(c_val));

		for (int i = 0; i < x_nRow; ++i) {
			one_hot_c[i][max_element(lookup_array[i].begin(), lookup_array[i].end()) - lookup_array[i].begin()] = 1;
		}

		return one_hot_c;
	}

public:
	F_Matrix getClustersCenter() {
		return v;
	}

	F_Matrix getFuzzyClusters() {
		return u;
	}

	void printClustersCenter() {
		cout << "Seq -> Clusters, Attributes Center Clusters = ,\n";
		for (int i = 0; i < c_val; ++i) {
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
		for (int i = 0; i < c_val; ++i)
			cout << i + 1 << (i == c_val - 1 ? "" : ", ")
			     << (i == c_val - 1 ? "\n" : "");

		for (int l = 0; l < x_nRow; ++l) {
			float m_val = *max_element(u[l].begin(), u[l].end());
			string num = to_string(l + 1);

			for (int i = num.length(); i < to_string(x_nRow).length(); ++i)
				num += " ";
			cout << num << " ";

			for (int j = 0; j < c_val; ++j) {
				cout << "| " << (mode_binary ? (u[l][j] == m_val ? "1" : " ") : to_string(u[l][j]))
				     << (j < c_val - 1 ? " " : " |");
			}
			cout << endl;

		}
	}

	F_Matrix build(F_Matrix data_x) {
		/*
		 * FCM Clustering Algorithm
		 */
		try {
			this->x = std::move(data_x);
			this->x_nRow = this->x.size();
			this->x_nCol = this->x[0].size();

			v.resize(c_val, F_Vec(x_nCol));
			p.resize(max_iter);

			if (randomize_u) {
				this->u.resize(x_nRow, F_Vec(c_val));
				URandomization();
			}
			for (int i = 0; i < max_iter; ++i) {
				printf("\n-------------- Iteration %i --------------\n", i + 1);
				computeV();
				computeP(i);
				updateU();

				float c = i > 0 ? abs(p[i] - p[i - 1]) : p[i] - 0;
				printf("p[%i]: %f\nminError: %f\nerror: %f\n", i, p[i], min_error, c);

				if ((i > 0 ? abs(p[i] - p[i - 1]) : p[i] - 0) < min_error) {
					cout << "\n### Stopping criterion reached, iteration stopped ###\n\n";
					break;
				} else continue;
			}
		} catch (exception &e) {
			cerr << e.what();
		}
		return FCMOneHot(u);
	}
};

FCMClustering::~FCMClustering() = default;

/**
 * Helper func.
 */

template<typename T>
void printMat(T &array,
              const std::string &delimiter_r = "\n",
              const std::string &delimiter_c = ", ",
              const std::string &caption = "-> Default Caption\n",
              bool numbered = true) {
	int i = 0;
	std::cout << caption;
	for (auto &r : array) { i++;
		cout << (numbered ? to_string(i) : "") << ". ";
		for (auto &c : r) std::cout << c << delimiter_c;
		std::cout << delimiter_r;
	}
}

int main() {
	F_Matrix x{{163, 8,  3218},
	           {159, 3,  1125},
	           {268, 13, 9249},
	           {375, 14, 253},
	           {359, 11, 1897},
	           {85,  4,  8540},
	           {79,  14, 4297},
	           {335, 8,  4874},
	           {295, 1,  583},
	           {385, 2,  932},
	           {300, 13, 2715},
	           {165, 12, 12321},
	           {238, 15, 8984},
	           {378, 1,  2696},
	           {5,   13, 801},
	           {232, 13, 600},
	           {368, 14, 364},
	           {155, 5,  3710},
	           {309, 11, 1190},
	           {203, 9,  2834},
	           {93,  6,  19205},
	           {86,  11, 719},
	           {162, 2,  626},
	           {268, 2,  8377},
	           {276, 4,  267}};
	F_Matrix u{{0.486761,  0.067977,  0.151048,  0.157281,  0.0924344, 0.0444984},
	           {0.050669,  0.0946507, 0.127477,  0.166282,  0.42753,   0.133392},
	           {0.0853756, 0.0748831, 0.060662,  0.556113,  0.161494,  0.0614723},
	           {0.165352,  0.0680026, 0.0862817, 0.137363,  0.162882,  0.380119},
	           {0.098484,  0.0532614, 0.163994,  0.0733051, 0.50749,   0.103466},
	           {0.114695,  0.554511,  0.124037,  0.0207482, 0.0489957, 0.137013},
	           {0.109011,  0.139406,  0.166386,  0.116917,  0.370336,  0.0979443},
	           {0.14969,   0.462126,  0.130438,  0.0990304, 0.0707099, 0.0880061},
	           {0.11808,   0.0677892, 0.165909,  0.107024,  0.090508,  0.450689},
	           {0.144607,  0.31051,   0.16383,   0.151262,  0.0954742, 0.134317},
	           {0.130191,  0.12225,   0.159527,  0.407726,  0.12037,   0.0599354},
	           {0.519564,  0.0169152, 0.126951,  0.159107,  0.118714,  0.0587487},
	           {0.0555743, 0.622892,  0.0492892, 0.0708761, 0.0475688, 0.1538},
	           {0.0755193, 0.0858123, 0.0805267, 0.07869,   0.628767,  0.0506851},
	           {0.0306632, 0.627365,  0.116046,  0.0503306, 0.0733287, 0.102266},
	           {0.123191,  0.13637,   0.145134,  0.106623,  0.355648,  0.133033},
	           {0.0455499, 0.0578942, 0.702569,  0.0499858, 0.111156,  0.0328455},
	           {0.515643,  0.0507381, 0.0893637, 0.103062,  0.166126,  0.075067},
	           {0.150848,  0.132406,  0.413281,  0.112403,  0.157545,  0.0335167},
	           {0.149174,  0.488511,  0.147102,  0.017908,  0.145349,  0.0519551},
	           {0.409809,  0.12338,   0.14916,   0.095286,  0.139094,  0.0832717},
	           {0.0459253, 0.502765,  0.0606942, 0.087284,  0.148777,  0.154555},
	           {0.0989073, 0.614513,  0.0301252, 0.147122,  0.0144144, 0.094918},
	           {0.119724,  0.40693,   0.0876784, 0.111081,  0.108288,  0.166298},
	           {0.137197,  0.0331271, 0.0997515, 0.0238895, 0.693851,  0.0121841}};
	FCMClustering m1(6, 2, pow(10, -6), 100, u);
	auto m1_oh = m1.build(x);

	cout << "Pengklasteran tipe pelanggan dengan matriks U terinisialisasi / matriks U makalah\n";
	m1.printClustersCenter();
	m1.printFuzzyClusters(true);
	printMat(m1_oh, "\n", ", ", "\nM1 One Hot Array: \n");

	cout << "\nPengklasteran tipe pelanggan dengan elemen random pada matriks U / matriks U sistem\n";
	FCMClustering m2(6, 2, pow(10, -6), 100);
	auto m2_oh = m2.build(x);
	m2.printClustersCenter();
	m2.printFuzzyClusters(true);
	printMat(m2_oh, "\n", ", ", "\nM2 One Hot Array: \n");
	system("pause");

	return 0;
}
