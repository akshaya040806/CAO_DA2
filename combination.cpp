#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>

template<typename T>
void computeFloydWarshallSeq(T* grid, size_t size)
{
    for (int mid = 0; mid < size; ++mid)
        for (int row = 0; row < size; ++row) {
            auto temp = grid[row * size + mid];
            for (int col = 0; col < size; ++col) {
                auto possible = temp + grid[mid * size + col];
                if (grid[row * size + col] > possible)
                    grid[row * size + col] = possible;
            }
        }
}

template<typename T>
void computeFloydWarshallPar(T* grid, size_t size, int threads)
{
    for (int mid = 0; mid < size; ++mid) {
#pragma omp parallel for num_threads(threads)
        for (int row = 0; row < size; ++row) {
            auto temp = grid[row * size + mid];
            for (int col = 0; col < size; ++col) {
                auto possible = temp + grid[mid * size + col];
                if (grid[row * size + col] > possible)
                    grid[row * size + col] = possible;
            }
        }
    }
}

class Timer {
    typedef std::chrono::steady_clock sys_clock;
    sys_clock::time_point begin;
public:
    Timer() : begin(sys_clock::now()) {}
    double elapsed() const {
        auto end = sys_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(end - begin).count();
    }
};

void runTest(size_t dimension, const int* threadList, int threadCount)
{
    float* base = new float[dimension * dimension];
    float* seqMatrix = new float[dimension * dimension];
    float* parMatrix = new float[dimension * dimension];
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist(0, 1);

    for (size_t i = 0; i < dimension * dimension; ++i)
        base[i] = dist(rng);

    for (size_t i = 0; i < dimension * dimension; ++i) {
        seqMatrix[i] = base[i];
        parMatrix[i] = base[i];
    }

    std::cout << "\nMatrix Size: " << dimension << "x" << dimension << std::endl;

    Timer t1;
    computeFloydWarshallSeq(seqMatrix, dimension);
    double seqTime = t1.elapsed();

    float sumSeq = 0;
    for (size_t i = 0; i < dimension * dimension; ++i)
        sumSeq += seqMatrix[i];

    for (int t = 0; t < threadCount; ++t) {
        int threads = threadList[t];
        for (size_t i = 0; i < dimension * dimension; ++i)
            parMatrix[i] = base[i];

        Timer t2;
        computeFloydWarshallPar(parMatrix, dimension, threads);
        double parTime = t2.elapsed();

        double speedup = seqTime / parTime;

        float sumPar = 0;
        for (size_t i = 0; i < dimension * dimension; ++i)
            sumPar += parMatrix[i];

        std::cout << "Threads: " << threads << "\n";
        std::cout << "Sequential Time: " << seqTime << " s\n";
        std::cout << "Parallel Time:   " << parTime << " s\n";
        std::cout << "Speedup: " << speedup << "x\n";
        std::cout << "Checksum (Seq): " << sumSeq << " | (Par): " << sumPar << "\n\n";
    }

    delete[] base;
    delete[] seqMatrix;
    delete[] parMatrix;
}

int main()
{
    int threadList[] = {1, 2, 4, 8, 16};
    int threadCount = sizeof(threadList) / sizeof(threadList[0]);

    std::cout << "Floyd–Warshall Sequential vs Parallel (OpenMP)\n";
    std::cout << "Testing with multiple thread counts...\n";

    runTest(500, threadList, threadCount);
    runTest(1000, threadList, threadCount);
    runTest(1500, threadList, threadCount);
    runTest(2000, threadList, threadCount);

    return 0;
}

