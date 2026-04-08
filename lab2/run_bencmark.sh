echo "=== OpenMP Matrix Multiplication Benchmark ==="
echo ""

cd build
cmake .. && make

echo ""
echo "Running benchmark..."
./benchmark_omp

echo ""
echo "Verification for 100x100 matrix..."
./matrix_mult_omp -t 100
python3 ../verify_omp.py verify_A.txt verify_B.txt verify_C.txt

echo ""
echo "Results:"
cat benchmark_results.csv