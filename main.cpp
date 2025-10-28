#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

bool is_prime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

int main(int argc, char **argv) {
    int limit = 200000000;
    if (argc > 1) {
        limit = std::atoi(argv[1]);
    }

    std::vector<int> thread_list = {1, 2, 4, 8};
    if (argc > 2) {
        thread_list.clear();
        for (int i = 2; i < argc; ++i) {
            int t = std::atoi(argv[i]);
            if (t > 0) {
                thread_list.push_back(t);
            }
        }
        if (thread_list.empty()) {
            thread_list = {1, 2, 4, 8};
        }
    }

    std::ofstream table("tiempos_openmp.csv");
    table << "modo,hilos,tiempo_seg,primos,limite\n";

    std::cout << "conteo de primos hasta " << limit << "\n";

    int total_seq = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int n = 2; n < limit; ++n) {
        if (is_prime(n)) total_seq++;
    }
    auto tf = std::chrono::high_resolution_clock::now();
    double time_seq = std::chrono::duration<double>(tf - t0).count();
    std::cout << "[seq] primos=" << total_seq << " tiempo=" << time_seq << "s\n";
    table << "secuencial,1," << time_seq << "," << total_seq << "," << limit << "\n";

    for (int threads : thread_list) {
        omp_set_num_threads(threads);

        int count_static = 0;
        auto a0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static) reduction(+:count_static)
        for (int n = 2; n < limit; ++n) {
            count_static += is_prime(n);
        }
        auto a1 = std::chrono::high_resolution_clock::now();
        double time_static = std::chrono::duration<double>(a1 - a0).count();
        std::cout << "[omp static] h=" << threads << " primos=" << count_static
                  << " tiempo=" << time_static << "s\n";
        table << "omp_static," << threads << "," << time_static << "," << count_static << "," << limit << "\n";

        int chunk_a = 8000;
        int count_static_chunk = 0;
        auto b0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static, chunk_a) reduction(+:count_static_chunk)
        for (int n = 2; n < limit; ++n) {
            count_static_chunk += is_prime(n);
        }
        auto b1 = std::chrono::high_resolution_clock::now();
        double time_static_chunk = std::chrono::duration<double>(b1 - b0).count();
        std::cout << "[omp static chunk] h=" << threads << " chunk=" << chunk_a
                  << " primos=" << count_static_chunk << " tiempo=" << time_static_chunk << "s\n";
        table << "omp_static_chunk," << threads << "," << time_static_chunk << "," << count_static_chunk << "," << limit << "\n";

        int chunk_b = 6000;
        int count_dynamic = 0;
        auto c0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, chunk_b) reduction(+:count_dynamic)
        for (int n = 2; n < limit; ++n) {
            count_dynamic += is_prime(n);
        }
        auto c1 = std::chrono::high_resolution_clock::now();
        double time_dynamic = std::chrono::duration<double>(c1 - c0).count();
        std::cout << "[omp dynamic] h=" << threads << " chunk=" << chunk_b
                  << " primos=" << count_dynamic << " tiempo=" << time_dynamic << "s\n";
        table << "omp_dynamic," << threads << "," << time_dynamic << "," << count_dynamic << "," << limit << "\n";

        int chunk_c = 20000;
        int count_guided = 0;
        auto d0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(guided, chunk_c) reduction(+:count_guided)
        for (int n = 2; n < limit; ++n) {
            count_guided += is_prime(n);
        }
        auto d1 = std::chrono::high_resolution_clock::now();
        double time_guided = std::chrono::duration<double>(d1 - d0).count();
        std::cout << "[omp guided] h=" << threads << " chunk=" << chunk_c
                  << " primos=" << count_guided << " tiempo=" << time_guided << "s\n";
        table << "omp_guided," << threads << "," << time_guided << "," << count_guided << "," << limit << "\n";
    }

    table.close();
    std::cout << "datos guardados en tiempos_openmp.csv\n";
    return 0;
}
