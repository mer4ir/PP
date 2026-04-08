#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

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

vector<vector<double>> multiplyParallelMPI(const vector<vector<double>>& A,
                                            const vector<vector<double>>& B,
                                            int rank, int size) {
    int n = A.size();
    
    int rows_per_proc = n / size;
    int remainder = n % size;
    
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    int offset = 0;
    for (int i = 0; i < rank; i++)
        offset += rows_per_proc + (i < remainder ? 1 : 0);
    
    vector<vector<double>> local_A(local_rows, vector<double>(n));
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < n; j++)
            local_A[i][j] = A[offset + i][j];
    
    vector<vector<double>> local_C(local_rows, vector<double>(n, 0.0));
    
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += local_A[i][k] * B[k][j];
            }
            local_C[i][j] = sum;
        }
    }
    
    vector<vector<double>> C(n, vector<double>(n));
    
    if (rank == 0) {
        for (int i = 0; i < local_rows; i++)
            for (int j = 0; j < n; j++)
                C[offset + i][j] = local_C[i][j];
        
        int current_offset = offset + local_rows;
        for (int p = 1; p < size; p++) {
            int p_rows = rows_per_proc + (p < remainder ? 1 : 0);
            
            vector<double> buffer(p_rows * n);
            MPI_Recv(buffer.data(), p_rows * n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int i = 0; i < p_rows; i++)
                for (int j = 0; j < n; j++)
                    C[current_offset + i][j] = buffer[i * n + j];
            
            current_offset += p_rows;
        }
    } else {
        vector<double> buffer(local_rows * n);
        for (int i = 0; i < local_rows; i++)
            for (int j = 0; j < n; j++)
                buffer[i * n + j] = local_C[i][j];
        
        MPI_Send(buffer.data(), local_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    return C;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 3 && argc != 4) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " -t N" << endl;
            cerr << "   or: " << argv[0] << " A.txt B.txt C.txt" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (string(argv[1]) == "-t") {
        int N = stoi(argv[2]);
        srand(time(nullptr) + rank);
        
        if (rank == 0) {
            cout << "\n=== MPI Matrix Multiplication ===" << endl;
            cout << "Size: " << N << "x" << N << endl;
            cout << "Processes: " << size << endl;
        }
        
        auto A = generateRandomMatrix(N);
        auto B = generateRandomMatrix(N);
        
        double start_time = MPI_Wtime();
        auto C = multiplyParallelMPI(A, B, rank, size);
        double end_time = MPI_Wtime();
        
        if (rank == 0) {
            long long ops = 2LL * N * N * N;
            double time_sec = end_time - start_time;
            double gflops = ops / time_sec / 1e9;
            
            cout << "Time: " << time_sec * 1000 << " ms" << endl;
            cout << "GFLOPS: " << fixed << setprecision(2) << gflops << endl;
            cout << "Operations: " << ops << endl;
            
            if (N <= 100) {
                auto C_seq = multiplySequential(A, B);
                double max_diff = 0.0;
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++)
                        max_diff = max(max_diff, abs(C[i][j] - C_seq[i][j]));
                
                if (max_diff < 1e-9)
                    cout << "Verification: PASSED" << endl;
                else
                    cout << "Verification: FAILED (max diff: " << max_diff << ")" << endl;
            }
        }
        
        MPI_Finalize();
        return 0;
    }
    
    string fileA = argv[1], fileB = argv[2], fileC = argv[3];
    
    int N;
    auto A = readMatrix(fileA, N);
    auto B = readMatrix(fileB, N);
    
    double start_time = MPI_Wtime();
    auto C = multiplyParallelMPI(A, B, rank, size);
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        writeMatrix(fileC, C);
        
        long long ops = 2LL * N * N * N;
        cout << "Size: " << N << "x" << N << endl;
        cout << "Processes: " << size << endl;
        cout << "Time: " << (end_time - start_time) * 1000 << " ms" << endl;
        cout << "Operations: " << ops << endl;
    }
    
    MPI_Finalize();
    return 0;
}