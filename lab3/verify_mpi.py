import numpy as np
import sys

def read_matrix(filename):
    with open(filename, 'r') as f:
        n = int(f.readline())
        return np.array([[float(x) for x in f.readline().split()] for _ in range(n)])

def main():
    if len(sys.argv) != 4:
        print("Usage: verify_mpi.py A.txt B.txt C.txt")
        return 1
    
    A = read_matrix(sys.argv[1])
    B = read_matrix(sys.argv[2])
    C_cpp = read_matrix(sys.argv[3])
    C_numpy = A @ B
    
    if np.allclose(C_cpp, C_numpy, atol=1e-10):
        print("VERIFICATION: PASSED")
        return 0
    else:
        print(f"VERIFICATION: FAILED (max diff: {np.max(np.abs(C_cpp - C_numpy)):.6e})")
        return 1

if __name__ == "__main__":
    sys.exit(main())