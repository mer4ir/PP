#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

#define BLOCK_SIZE 16

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

int main() {
    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};
    vector<int> block_sizes = {8, 16, 32};
    
    cout << "Size,BlockSize,CPU_Time_ms,GPU_Time_ms,Speedup" << endl;
    
    for (int N : sizes) {
        srand(42);
        auto A = generateRandomMatrix(N);
        auto B = generateRandomMatrix(N);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto C_cpu = multiplyCPU(A, B);
        auto end = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        
        for (int bs : block_sizes) {
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
            
            cudaEvent_t gpu_start, gpu_stop;
            cudaEventCreate(&gpu_start);
            cudaEventCreate(&gpu_stop);
            
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
            
            cout << N << "," << bs << "," << cpu_time << "," << gpu_time_ms << "," << speedup << endl;
        }
    }
    
    return 0;
}