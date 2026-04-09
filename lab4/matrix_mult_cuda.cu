#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

#define BLOCK_SIZE 16

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

void writeResults(const string& filename, int size, double cpu_time, double gpu_time, double speedup, int block_size) {
    ofstream file(filename, ios::app);
    file << size << "," << block_size << "," << cpu_time << "," << gpu_time << "," << speedup << "\n";
    file.close();
}

vector<vector<double>> generateRandomMatrix(int size) {
    vector<vector<double>> matrix(size, vector<double>(size));
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            matrix[i][j] = rand() % 10 + 1;
    return matrix;
}

vector<vector<double>> multiplyCPU(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

__global__ void matrixMulKernel(double* A, double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

vector<vector<double>> multiplyGPU(const vector<vector<double>>& A, const vector<vector<double>>& B, int block_size) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    
    size_t bytes = n * n * sizeof(double);
    
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    vector<double> A_flat(n * n);
    vector<double> B_flat(n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_flat[i * n + j] = A[i][j];
            B_flat[i * n + j] = B[i][j];
        }
    }
    
    cudaMemcpy(d_A, A_flat.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_flat.data(), bytes, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(block_size, block_size);
    dim3 numBlocks((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);
    
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    
    vector<double> C_flat(n * n);
    cudaMemcpy(C_flat.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = C_flat[i * n + j];
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return C;
}

int main(int argc, char* argv[]) {
    if (argc == 5) {
        string fileA = argv[1], fileB = argv[2], fileC = argv[3];
        int block_size = stoi(argv[4]);
        
        int N;
        auto A = readMatrix(fileA, N);
        auto B = readMatrix(fileB, N);
        
        auto start = high_resolution_clock::now();
        auto C = multiplyGPU(A, B, block_size);
        auto end = high_resolution_clock::now();
        double gpu_time = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        writeMatrix(fileC, C);
        
        cout << "Size: " << N << "x" << N << endl;
        cout << "Block Size: " << block_size << "x" << block_size << endl;
        cout << "GPU Time: " << gpu_time << " ms" << endl;
        
        return 0;
    }
    
    if (argc == 3 && string(argv[1]) == "-t") {
        int N = stoi(argv[2]);
        srand(time(nullptr));
        
        auto A = generateRandomMatrix(N);
        auto B = generateRandomMatrix(N);
        
        cout << "\n=== CUDA Matrix Multiplication ===" << endl;
        cout << "Size: " << N << "x" << N << endl;
        
        auto start = high_resolution_clock::now();
        auto C_cpu = multiplyCPU(A, B);
        auto end = high_resolution_clock::now();
        double cpu_time = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        vector<int> block_sizes = {8, 16, 32};
        
        cout << "\nCPU Time: " << cpu_time << " ms" << endl;
        cout << "\nBlock Size | GPU Time (ms) | Speedup" << endl;
        cout << "-----------|---------------|---------" << endl;
        
        for (int bs : block_sizes) {
            cudaEvent_t gpu_start, gpu_stop;
            cudaEventCreate(&gpu_start);
            cudaEventCreate(&gpu_stop);
            
            size_t bytes = N * N * sizeof(double);
            double *d_A, *d_B, *d_C;
            cudaMalloc(&d_A, bytes);
            cudaMalloc(&d_B, bytes);
            cudaMalloc(&d_C, bytes);
            
            vector<double> A_flat(N * N);
            vector<double> B_flat(N * N);
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    A_flat[i * N + j] = A[i][j];
                    B_flat[i * N + j] = B[i][j];
                }
            }
            
            cudaMemcpy(d_A, A_flat.data(), bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, B_flat.data(), bytes, cudaMemcpyHostToDevice);
            
            dim3 threadsPerBlock(bs, bs);
            dim3 numBlocks((N + bs - 1) / bs, (N + bs - 1) / bs);
            
            cudaEventRecord(gpu_start);
            matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
            cudaEventRecord(gpu_stop);
            cudaEventSynchronize(gpu_stop);
            
            float gpu_time_ms;
            cudaEventElapsedTime(&gpu_time_ms, gpu_start, gpu_stop);
            
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            
            double speedup = cpu_time / gpu_time_ms;
            
            cout << "   " << bs << "x" << bs << "     |     " << gpu_time_ms << "     |   " << speedup << "x" << endl;
            
            if (N <= 100 && bs == 16) {
                writeMatrix("verify_A.txt", A);
                writeMatrix("verify_B.txt", B);
                
                vector<double> C_flat(N * N);
                cudaMemcpy(C_flat.data(), d_C, bytes, cudaMemcpyDeviceToHost);
                vector<vector<double>> C_gpu(N, vector<double>(N));
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++)
                        C_gpu[i][j] = C_flat[i * N + j];
                writeMatrix("verify_C.txt", C_gpu);
            }
        }
        
        return 0;
    }
    
    cerr << "Usage:" << endl;
    cerr << "  " << argv[0] << " -t N" << endl;
    cerr << "  " << argv[0] << " A.txt B.txt C.txt block_size" << endl;
    return 1;
}