#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <mpi.h>

using namespace std;

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
    
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                local_C[i][j] += local_A[i][k] * B[k][j];
    
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
    
    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};
    
    ofstream csv;
    if (rank == 0) {
        csv.open("mpi_results.csv");
        csv << "Processes,Size,Time_ms,GFLOPS,Speedup,Efficiency\n";
        cout << "=== MPI Benchmark Results ===\n" << endl;
    }
    
    for (int N : sizes) {
        srand(42 + rank);
        auto A = generateRandomMatrix(N);
        auto B = generateRandomMatrix(N);
        
        double seq_time = 0;
        if (rank == 0) {
            cout << "Testing size " << N << "x" << N << "..." << endl;
            double start = MPI_Wtime();
            auto C_seq = multiplySequential(A, B);
            double end = MPI_Wtime();
            seq_time = end - start;
        }
        MPI_Bcast(&seq_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        double start = MPI_Wtime();
        auto C_par = multiplyParallelMPI(A, B, rank, size);
        double end = MPI_Wtime();
        double par_time = end - start;
        
        double max_par_time = par_time;
        MPI_Reduce(&par_time, &max_par_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            long long ops = 2LL * N * N * N;
            double speedup = seq_time / max_par_time;
            double efficiency = (speedup / size) * 100;
            double gflops = ops / max_par_time / 1e9;
            
            cout << "  Processes: " << size << endl;
            cout << "  Time: " << max_par_time * 1000 << " ms" << endl;
            cout << "  Speedup: " << fixed << setprecision(2) << speedup << "x" << endl;
            cout << "  Efficiency: " << efficiency << "%" << endl;
            cout << "  GFLOPS: " << gflops << endl;
            cout << endl;
            
            csv << size << "," << N << "," << max_par_time * 1000 << ","
                << fixed << setprecision(2) << gflops << ","
                << speedup << "," << efficiency << "\n";
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        csv.close();
        cout << "\nResults saved to mpi_results.csv" << endl;
    }
    
    MPI_Finalize();
    return 0;
}