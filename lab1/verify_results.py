import numpy as np
import sys

def read_matrix(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        size = int(lines[0].strip())
        
        matrix = []
        for i in range(size):
            row = list(map(float, lines[i+1].strip().split()))
            matrix.append(row)
        
        return np.array(matrix)
    
    except FileNotFoundError:
        print(f"Ошибка: файл {filename} не найден")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")
        sys.exit(1)

def verify_multiplication(file_a, file_b, file_c, tolerance=1e-10):
    print("\nАВТОМАТИЧЕСКАЯ ВЕРИФИКАЦИЯ РЕЗУЛЬТАТОВ")
    
    A = read_matrix(file_a)
    B = read_matrix(file_b)
    C_cpp = read_matrix(file_c)
    
    print(f"   Матрица A: {A.shape[0]}x{A.shape[1]}")
    print(f"   Матрица B: {B.shape[0]}x{B.shape[1]}")
    print(f"   Матрица C: {C_cpp.shape[0]}x{C_cpp.shape[1]}")
    
    C_numpy = np.dot(A, B)
    
    difference = np.abs(C_cpp - C_numpy)
    max_diff = np.max(difference)
    mean_diff = np.mean(difference)
    
    print("\nРЕЗУЛЬТАТЫ СРАВНЕНИЯ:")
    print(f"   Максимальная разница: {max_diff:.6e}")
    print(f"   Средняя разница: {mean_diff:.6e}")
    
    if max_diff < tolerance:
        print("\nВЕРИФИКАЦИЯ ПРОЙДЕНА: результаты совпадают!")
        print(f"   (погрешность в пределах {tolerance})")
        return True
    else:
        print("\nВЕРИФИКАЦИЯ НЕ ПРОЙДЕНА: результаты отличаются!")
        print(f"   (погрешность превышает {tolerance})")
        
        print("\nПервые элементы матриц:")
        n = min(3, A.shape[0])
        print("   C++ результат:")
        print(C_cpp[:n, :n])
        print("   NumPy результат:")
        print(C_numpy[:n, :n])
        
        return False

def main():
    if len(sys.argv) != 4:
        print("ИСПОЛЬЗОВАНИЕ:")
        print(f"  {sys.argv[0]} matrix_A.txt matrix_B.txt matrix_C.txt")
        print("\nГде:")
        print("  matrix_A.txt - файл с первой матрицей")
        print("  matrix_B.txt - файл со второй матрицей")
        print("  matrix_C.txt - файл с результатом умножения (из C++ программы)")
        sys.exit(1)
    
    file_a = sys.argv[1]
    file_b = sys.argv[2]
    file_c = sys.argv[3]
    
    success = verify_multiplication(file_a, file_b, file_c)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()