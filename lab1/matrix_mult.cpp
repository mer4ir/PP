#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace chrono;

vector<vector<double>> readMatrix(const string& filename, int& size) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка: не удалось открыть файл " << filename << endl;
        exit(1);
    }
    
    file >> size;
    vector<vector<double>> matrix(size, vector<double>(size));
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            file >> matrix[i][j];
        }
    }
    
    file.close();
    return matrix;
}

void writeMatrix(const string& filename, const vector<vector<double>>& matrix) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка: не удалось создать файл " << filename << endl;
        exit(1);
    }
    
    int size = matrix.size();
    file << size << endl;
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            file << fixed << setprecision(6) << matrix[i][j] << " ";
        }
        file << endl;
    }
    
    file.close();
}

void writeResults(const string& filename, int matrixSize, long long durationMicroseconds) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка: не удалось создать файл результатов " << filename << endl;
        exit(1);
    }
    
    double durationMs = durationMicroseconds / 1000.0;
    long long operations = 2LL * matrixSize * matrixSize * matrixSize;
    double mflops = operations / (durationMicroseconds / 1e6) / 1e6;
    
    file << "\nРЕЗУЛЬТАТЫ УМНОЖЕНИЯ МАТРИЦ\n";
    file << "Размер матриц: " << matrixSize << " x " << matrixSize << "\n";
    file << "\nОБЪЕМ ЗАДАЧИ:\n";
    file << "  Элементов в матрице: " << matrixSize * matrixSize << "\n";
    file << "  Всего элементов (A+B+C): " << 3 * matrixSize * matrixSize << "\n";
    file << "  Память под данные: " << fixed << setprecision(2) 
         << (3.0 * matrixSize * matrixSize * sizeof(double)) / 1024.0 << " КБ\n";
    file << "  Операций: " << operations << " (умножений + сложений)\n";
    file << "\nВРЕМЯ ВЫПОЛНЕНИЯ:\n";
    file << "  " << durationMicroseconds << " мкс\n";
    file << "  " << fixed << setprecision(3) << durationMs << " мс\n";
    file << "  " << fixed << setprecision(6) << durationMs / 1000.0 << " с\n";
    file << "\nПРОИЗВОДИТЕЛЬНОСТЬ:\n";
    file << "  " << fixed << setprecision(2) << mflops << " MFLOPS\n";
    
    file.close();
}

vector<vector<double>> multiplyMatrices(const vector<vector<double>>& A, 
                                        const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return C;
}

vector<vector<double>> generateRandomMatrix(int size) {
    vector<vector<double>> matrix(size, vector<double>(size));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = rand() % 10;
        }
    }
    return matrix;
}

int main(int argc, char* argv[]) {
    string fileA, fileB, fileC, fileResults;
    int matrixSize = 0;
    bool useRandom = false;
    
    if (argc == 1) {
        useRandom = true;
        matrixSize = 5;
        fileResults = "results.txt";
        cout << "Режим: случайные матрицы " << matrixSize << "x" << matrixSize << endl;
        
    } else if (argc == 5) {
        fileA = argv[1];
        fileB = argv[2];
        fileC = argv[3];
        fileResults = argv[4];
        cout << "Режим: умножение матриц из файлов" << endl;
        
    } else if (argc == 4 && string(argv[1]) == "-r") {
        useRandom = true;
        matrixSize = stoi(argv[2]);
        fileResults = argv[3];
        cout << "Режим: случайные матрицы " << matrixSize << "x" << matrixSize << endl;
        
    } else {
        cerr << "\nИСПОЛЬЗОВАНИЕ:\n";
        cerr << "  " << argv[0] << "                     # случайные матрицы 5x5\n";
        cerr << "  " << argv[0] << " -r N file.txt       # случайные матрицы NxN, результаты в file.txt\n";
        cerr << "  " << argv[0] << " A.txt B.txt C.txt R.txt  # из файлов A и B, результат в C, отчет в R\n";
        return 1;
    }
    
    vector<vector<double>> A, B;
    
    if (useRandom) {
        srand(time(nullptr));
        A = generateRandomMatrix(matrixSize);
        B = generateRandomMatrix(matrixSize);
        
        writeMatrix("matrix_A.txt", A);
        writeMatrix("matrix_B.txt", B);
        cout << "Сгенерированы матрицы:" << endl;
        cout << "  matrix_A.txt" << endl;
        cout << "  matrix_B.txt" << endl;
    } else {
        int sizeA, sizeB;
        A = readMatrix(fileA, sizeA);
        B = readMatrix(fileB, sizeB);
        
        if (sizeA != sizeB) {
            cerr << "Ошибка: матрицы разного размера!" << endl;
            return 1;
        }
        matrixSize = sizeA;
    }
    
    auto start = high_resolution_clock::now();
    vector<vector<double>> C = multiplyMatrices(A, B);
    auto end = high_resolution_clock::now();
    
    auto duration = duration_cast<microseconds>(end - start);
    
    if (useRandom) {
        writeMatrix("matrix_C.txt", C);
        cout << "Результат умножения записан в matrix_C.txt" << endl;
    } else {
        writeMatrix(fileC, C);
        cout << "Результат: " << fileC << endl;
    }
    
    writeResults(fileResults, matrixSize, duration.count());
    cout << "Отчет записан в " << fileResults << endl;
    
    cout << "\nКРАТКИЙ ОТЧЕТ" << endl;
    cout << "  Размер: " << matrixSize << "x" << matrixSize << endl;
    cout << "  Время: " << duration.count() << " мкс (" 
         << fixed << setprecision(3) << duration.count() / 1000.0 << " мс)" << endl;
    cout << "  Подробности в файле: " << fileResults << endl;
    
    return 0;
}