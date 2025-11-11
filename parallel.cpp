#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>

template<typename T>
void computeFloydWarshall(T* grid, size_t size, int threadCount = 0)
{
    for (int mid = 0; mid < size; ++mid) {
#pragma omp parallel for num_threads(threadCount)
        for (int row = 0; row < size; ++row) {
            auto temp = grid[row * size + mid];
            for (int col = 0; col < size; ++col) {
                auto possible = temp + grid[mid * size + col];
                if (grid[row * size + col] > possible) {
                    grid[row * size + col] = possible;
                }
            }
        }
    }
}

class Timer {
    typedef std::chrono::steady_clock sys_clock;
    sys_clock::time_point begin;
public:
    Timer() : begin(sys_clock::now()) {}
    ~Timer() {
        auto end = sys_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        std::cout << "Time taken: " << duration << " us";
    }
};

void runTest(size_t dimension)
{
    float* mainMatrix = new float[dimension * dimension];
    float* backupMatrix = new float[dimension * dimension];
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist(0, 1);

    for (size_t i = 0; i < dimension * dimension; ++i) backupMatrix[i] = dist(rng);

    std::cout << "\nTesting Matrix Size: " << dimension << "x" << dimension << std::endl;
    int totalThreads = omp_get_max_threads() * 2;

    for (int threads = 0; threads <= totalThreads; ++threads) {
        std::cout << threads << " Threads\n";
        for (int run = 0; run < 4; ++run) {
            for (size_t i = 0; i < dimension * dimension; ++i)
                mainMatrix[i] = backupMatrix[i];

            {
                Timer timer;
                computeFloydWarshall(mainMatrix, dimension, threads);
            }

            float checkValue = 0;
            for (size_t i = 0; i < dimension * dimension; ++i)
                checkValue += i * mainMatrix[i] / (dimension * dimension);
            std::cout << " | Checksum: " << checkValue << std::endl;
        }
    }

    delete[] mainMatrix;
    delete[] backupMatrix;
}

int main()
{
    runTest(100);
    runTest(200);
    runTest(1000);
    return 0;
}

