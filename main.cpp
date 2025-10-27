#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

bool es_primo(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

int main(int argc, char **argv) {
    int limite = 200000000;
    if (argc > 1) {
        limite = std::atoi(argv[1]);
        if (limite < 10) {
            std::cerr << "limite muy chico, usando 1000000\n";
            limite = 1000000;
        }
    }

    std::vector<int> lista_hilos = {1, 2, 4, 8};
    if (argc > 2) {
        lista_hilos.clear();
        for (int i = 2; i < argc; ++i) {
            int t = std::atoi(argv[i]);
            if (t > 0) {
                lista_hilos.push_back(t);
            }
        }
        if (lista_hilos.empty()) {
            lista_hilos = {1, 2, 4, 8};
        }
    }

    std::ofstream tabla("tiempos_openmp.csv");
    tabla << "modo,hilos,tiempo_seg,primos,limite\n";

    std::cout << "conteo de primos hasta " << limite << "\n";

    int total_seq = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int n = 2; n < limite; ++n) {
        if (es_primo(n)) total_seq++;
    }
    auto tf = std::chrono::high_resolution_clock::now();
    double tiempo_seq = std::chrono::duration<double>(tf - t0).count();
    std::cout << "[seq] primos=" << total_seq << " tiempo=" << tiempo_seq << "s\n";
    tabla << "secuencial,1," << tiempo_seq << "," << total_seq << "," << limite << "\n";

    for (int hilos : lista_hilos) {
        omp_set_num_threads(hilos);

        int cuenta_static = 0;
        auto a0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static) reduction(+:cuenta_static)
        for (int n = 2; n < limite; ++n) {
            cuenta_static += es_primo(n);
        }
        auto a1 = std::chrono::high_resolution_clock::now();
        double tiempo_static = std::chrono::duration<double>(a1 - a0).count();
        std::cout << "[omp static] h=" << hilos << " primos=" << cuenta_static
                  << " tiempo=" << tiempo_static << "s\n";
        tabla << "omp_static," << hilos << "," << tiempo_static << "," << cuenta_static << "," << limite << "\n";

        int chunk_a = 8000;
        int cuenta_static_chunk = 0;
        auto b0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static, chunk_a) reduction(+:cuenta_static_chunk)
        for (int n = 2; n < limite; ++n) {
            cuenta_static_chunk += es_primo(n);
        }
        auto b1 = std::chrono::high_resolution_clock::now();
        double tiempo_static_chunk = std::chrono::duration<double>(b1 - b0).count();
        std::cout << "[omp static chunk] h=" << hilos << " chunk=" << chunk_a
                  << " primos=" << cuenta_static_chunk << " tiempo=" << tiempo_static_chunk << "s\n";
        tabla << "omp_static_chunk," << hilos << "," << tiempo_static_chunk << "," << cuenta_static_chunk << "," << limite << "\n";

        int chunk_b = 6000;
        int cuenta_dynamic = 0;
        auto c0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, chunk_b) reduction(+:cuenta_dynamic)
        for (int n = 2; n < limite; ++n) {
            cuenta_dynamic += es_primo(n);
        }
        auto c1 = std::chrono::high_resolution_clock::now();
        double tiempo_dynamic = std::chrono::duration<double>(c1 - c0).count();
        std::cout << "[omp dynamic] h=" << hilos << " chunk=" << chunk_b
                  << " primos=" << cuenta_dynamic << " tiempo=" << tiempo_dynamic << "s\n";
        tabla << "omp_dynamic," << hilos << "," << tiempo_dynamic << "," << cuenta_dynamic << "," << limite << "\n";

        int chunk_c = 20000;
        int cuenta_guided = 0;
        auto d0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(guided, chunk_c) reduction(+:cuenta_guided)
        for (int n = 2; n < limite; ++n) {
            cuenta_guided += es_primo(n);
        }
        auto d1 = std::chrono::high_resolution_clock::now();
        double tiempo_guided = std::chrono::duration<double>(d1 - d0).count();
        std::cout << "[omp guided] h=" << hilos << " chunk=" << chunk_c
                  << " primos=" << cuenta_guided << " tiempo=" << tiempo_guided << "s\n";
        tabla << "omp_guided," << hilos << "," << tiempo_guided << "," << cuenta_guided << "," << limite << "\n";
    }

    tabla.close();
    std::cout << "datos guardados en tiempos_openmp.csv\n";
    return 0;
}
