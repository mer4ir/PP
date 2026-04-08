#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <omp.h>

using namespace std;
using namespace chrono;

vector<vector<double>> generateRandomMatrix(int size) {
    vector<vector<double>> matrix(size, vector<double>(size));
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            matrix[i][j] = rand() % 10 + 1;
    return matrix;
}

vector<vector<double>> multiplySequential(const vector<vector<double>>& A, 
                                           const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

vector<vector<double>> multiplyParallel(const vector<vector<double>>& A, 
                                         const vector<vector<double>>& B,
                                         int numThreads) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    omp_set_num_threads(numThreads);
    
    #pragma omp parallel for collapse(2) schedule(static)
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

int main() {
    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};
    vector<int> threads = {1, 2, 4, 8};
    int maxThreads = omp_get_max_threads();
    
    cout << "System max threads: " << maxThreads << endl;
    
    ofstream csv("benchmark_results.csv");
    csv << "Size,Threads,Time_us,Time_ms,Operations,GFLOPS,Speedup,Efficiency\n";
    
    for (int size : sizes) {
        cout << "\nTesting size " << size << "x" << size << endl;
        
        srand(42);
        auto A = generateRandomMatrix(size);
        auto B = generateRandomMatrix(size);
        
        long long ops = 2LL * size * size * size;
        double seqTime = 0;
        
        auto start = high_resolution_clock::now();
        auto C_seq = multiplySequential(A, B);
        auto end = high_resolution_clock::now();
        seqTime = duration_cast<microseconds>(end - start).count();
        cout << "  Sequential: " << seqTime / 1000.0 << " ms" << endl;
        
        for (int t : threads) {
            if (t > maxThreads) continue;
            
            const int runs = 3;
            double totalTime = 0;
            
            for (int r = 0; r < runs; r++) {
                auto start = high_resolution_clock::now();
                auto C_par = multiplyParallel(A, B, t);
                auto end = high_resolution_clock::now();
                totalTime += duration_cast<microseconds>(end - start).count();
            }
            
            double avgTime = totalTime / runs;
            double speedup = seqTime / avgTime;
            double efficiency = speedup / t;
            double gflops = ops / (avgTime / 1e6) / 1e9;
            
            cout << "  Threads " << t << ": " << avgTime / 1000.0 
                 << " ms, speedup=" << fixed << setprecision(2) << speedup << "x" << endl;
            
            csv << size << "," << t << "," << avgTime << "," << avgTime/1000.0 << ","
                << ops << "," << gflops << "," << speedup << "," << efficiency << "\n";
        }
    }
    
    csv.close();
    cout << "\nResults saved to benchmark_results.csv" << endl;
    
    return 0;
}