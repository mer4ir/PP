echo "=== CUDA Matrix Multiplication Benchmark ==="

mkdir -p build && cd build
cmake .. && make

for size in 200 400 800 1200 1600 2000; do
    echo ""
    echo "Testing size $size x $size"
    ./matrix_mult_cuda -t $size
done

echo ""
echo "Benchmark complete. Results above."