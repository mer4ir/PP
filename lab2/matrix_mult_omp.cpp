#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace chrono;

vector<vector<double>> readMatrix(const string& filename, int& size) {
    ifstream file(filename);
    file >> size;
    vector<vector<double>> matrix(size, vector<double>(size));
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            file >> matrix[i][j];
    return matrix;
}

void writeMatrix(const string& filename, const vector<vector<double>>& matrix) {
    ofstream file(filename);
    int size = matrix.size();
    file << size << endl;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            file << fixed << setprecision(6) << matrix[i][j] << " ";
        file << endl;
    }
}

vector<vector<double>> multiplyParallel(const vector<vector<double>>& A, 
                                         const vector<vector<double>>& B,
                                         int numThreads) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    
#ifdef _OPENMP
    omp_set_num_threads(numThreads);
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}

vector<vector<double>> generateRandomMatrix(int size) {
    vector<vector<double>> matrix(size, vector<double>(size));
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            matrix[i][j] = rand() % 10 + 1;
    return matrix;
}

int main(int argc, char* argv[]) {
    if (argc != 5 && argc != 2) {
        cerr << "Usage: " << argv[0] << " A.txt B.txt C.txt threads" << endl;
        cerr << "   or: " << argv[0] << " -t size" << endl;
        return 1;
    }
    
    if (string(argv[1]) == "-t") {
        int size = stoi(argv[2]);
        srand(time(nullptr));
        auto A = generateRandomMatrix(size);
        auto B = generateRandomMatrix(size);
        
#ifdef _OPENMP
        int maxThreads = omp_get_max_threads();
        cout << "Max threads: " << maxThreads << endl;
        
        for (int t : {1, 2, 4, 8}) {
            if (t > maxThreads) continue;
#else
        {
            int t = 1;
#endif
            
            auto start = high_resolution_clock::now();
            auto C = multiplyParallel(A, B, t);
            auto end = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end - start);
            
            long long ops = 2LL * size * size * size;
            double gflops = ops / (duration.count() / 1e6) / 1e9;
            
            cout << "Threads=" << t << " Time=" << duration.count() 
                 << " us GFLOPS=" << fixed << setprecision(2) << gflops << endl;
#ifdef _OPENMP
        }
#endif
        return 0;
    }
    
    string fileA = argv[1], fileB = argv[2], fileC = argv[3];
    int threads = stoi(argv[4]);
    
    int sizeA, sizeB;
    auto A = readMatrix(fileA, sizeA);
    auto B = readMatrix(fileB, sizeB);
    
    if (sizeA != sizeB) {
        cerr << "Matrices must be same size" << endl;
        return 1;
    }
    
    auto start = high_resolution_clock::now();
    auto C = multiplyParallel(A, B, threads);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    
    writeMatrix(fileC, C);
    
    long long ops = 2LL * sizeA * sizeA * sizeA;
    cout << "Size: " << sizeA << "x" << sizeA << endl;
    cout << "Threads: " << threads << endl;
    cout << "Time: " << duration.count() << " us (" 
         << duration.count() / 1000.0 << " ms)" << endl;
    cout << "Operations: " << ops << endl;
    
    return 0;
}