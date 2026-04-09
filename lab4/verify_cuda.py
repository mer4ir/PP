import numpy as np
import sys

def read_matrix(filename):
    with open(filename, 'r') as f:
        n = int(f.readline())
        return np.array([[float(x) for x in f.readline().split()] for _ in range(n)])

def main():
    if len(sys.argv) != 4:
        print("Usage: verify_cuda.py A.txt B.txt C.txt")
        return 1
    
    A = read_matrix(sys.argv[1])
    B = read_matrix(sys.argv[2])
    C_cuda = read_matrix(sys.argv[3])
    C_numpy = np.dot(A, B)
    
    print("\n=== VERIFICATION ===")
    print(f"Matrix size: {A.shape[0]}x{A.shape[1]}")
    
    if np.allclose(C_cuda, C_numpy, atol=1e-8):
        print("RESULT: PASSED")
        return 0
    else:
        max_diff = np.max(np.abs(C_cuda - C_numpy))
        print(f"RESULT: FAILED (max difference: {max_diff:.6e})")
        return 1

if __name__ == "__main__":
    sys.exit(main())