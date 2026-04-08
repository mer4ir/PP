echo "=== MPI Benchmark ==="

mkdir -p build && cd build
cmake .. && make

rm -f mpi_results.csv

for procs in 1 2 4 8; do
    echo "Running with $procs processes..."
    mpirun -np $procs ./benchmark_mpi
done

echo ""
echo "Results saved to mpi_results.csv"
cat mpi_results.csv