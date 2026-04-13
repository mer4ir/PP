#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

vector<vector<double> > readMatrix(const string& filename, int& size) {
    ifstream file(filename.c_str());
    file >> size;
    vector<vector<double> > matrix(size, vector<double>(size));
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            file >> matrix[i][j];
    return matrix;
}

void writeMatrix(const string& filename, const vector<vector<double> >& matrix) {
    ofstream file(filename.c_str());
    int size = matrix.size();
    file << size << endl;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            file << matrix[i][j] << " ";
        file << endl;
    }
}

vector<vector<double> > generateRandomMatrix(int size) {
    vector<vector<double> > matrix(size, vector<double>(size));
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            matrix[i][j] = rand() % 10 + 1;
    return matrix;
}

vector<vector<double> > multiplySequential(const vector<vector<double> >& A, 
                                            const vector<vector<double> >& B) {
    int n = A.size();
    vector<vector<double> > C(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

vector<vector<double> > multiplyParallelMPI(const vector<vector<double> >& A,
                                             const vector<vector<double> >& B,
                                             int rank, int size) {
    int n = A.size();
    int rows_per_proc = n / size;
    int remainder = n % size;
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    
    int offset = 0;
    for (int i = 0; i < rank; i++)
        offset += rows_per_proc + (i < remainder ? 1 : 0);
    
    vector<vector<double> > local_A(local_rows, vector<double>(n));
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < n; j++)
            local_A[i][j] = A[offset + i][j];
    
    vector<vector<double> > local_C(local_rows, vector<double>(n, 0.0));
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                local_C[i][j] += local_A[i][k] * B[k][j];
    
    vector<vector<double> > C(n, vector<double>(n));
    
    if (rank == 0) {
        for (int i = 0; i < local_rows; i++)
            for (int j = 0; j < n; j++)
                C[offset + i][j] = local_C[i][j];
        
        int current_offset = offset + local_rows;
        for (int p = 1; p < size; p++) {
            int p_rows = rows_per_proc + (p < remainder ? 1 : 0);
            vector<double> buffer(p_rows * n);
            MPI_Recv(&buffer[0], p_rows * n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
        MPI_Send(&buffer[0], local_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    return C;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc == 4 && string(argv[1]) != "-t") {
        string fileA = argv[1], fileB = argv[2], fileC = argv[3];
        int N;
        vector<vector<double> > A = readMatrix(fileA, N);
        vector<vector<double> > B = readMatrix(fileB, N);
        
        double start_time = MPI_Wtime();
        vector<vector<double> > C = multiplyParallelMPI(A, B, rank, size);
        double end_time = MPI_Wtime();
        
        if (rank == 0) {
            writeMatrix(fileC, C);
            cout << "Time: " << (end_time - start_time) * 1000 << " ms" << endl;
        }
    }
    else if (argc == 3 && string(argv[1]) == "-t") {
        int N = atoi(argv[2]);
        srand(time(NULL) + rank);
        
        vector<vector<double> > A = generateRandomMatrix(N);
        vector<vector<double> > B = generateRandomMatrix(N);
        
        double seq_time = 0;
        if (rank == 0) {
            double start = MPI_Wtime();
            vector<vector<double> > C_seq = multiplySequential(A, B);
            double end = MPI_Wtime();
            seq_time = (end - start) * 1000;
            cout << "Size: " << N << " x " << N << endl;
            cout << "Sequential: " << seq_time << " ms" << endl;
        }
        MPI_Bcast(&seq_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        double start_time = MPI_Wtime();
        vector<vector<double> > C_par = multiplyParallelMPI(A, B, rank, size);
        double end_time = MPI_Wtime();
        double par_time = (end_time - start_time) * 1000;
        
        double max_par_time = par_time;
        MPI_Reduce(&par_time, &max_par_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            double speedup = seq_time / max_par_time;
            double efficiency = (speedup / size) * 100;
            long long ops = 2LL * N * N * N;
            double gflops = ops / (max_par_time / 1000.0) / 1e9;
            
            cout << "MPI (" << size << " processes): " << max_par_time << " ms" << endl;
            cout << "Speedup: " << speedup << "x" << endl;
            cout << "Efficiency: " << efficiency << "%" << endl;
            cout << "GFLOPS: " << gflops << endl;
            
            if (N <= 100) {
                writeMatrix("verify_A.txt", A);
                writeMatrix("verify_B.txt", B);
                writeMatrix("verify_C.txt", C_par);
            }
        }
    }
    else {
        if (rank == 0) {
            cout << "Usage:" << endl;
            cout << "  mpiexec -n N ./matrix_mult_mpi -t size" << endl;
            cout << "  mpiexec -n N ./matrix_mult_mpi A.txt B.txt C.txt" << endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}