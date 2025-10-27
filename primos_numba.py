import time
import csv
from pathlib import Path

from numba import njit, prange, set_num_threads


@njit(parallel=True)
def contar_primos(limite):
    total = 0
    for n in prange(2, limite):
        es = True
        if n == 2:
            es = True
        elif n < 2:
            es = False
        elif n % 2 == 0:
            es = False
        else:
            i = 3
            while i * i <= n and es:
                if n % i == 0:
                    es = False
                i += 2
        if es:
            total += 1
    return total


def medir(limite, hilos):
    set_num_threads(hilos)
    t0 = time.perf_counter()
    total = contar_primos(limite)
    t1 = time.perf_counter()
    return total, t1 - t0


def main():
    limite = 200_000_000
    lista_hilos = [1, 2, 4, 8]

    # hacer un primer llamado cortito para que numba compile
    set_num_threads(1)
    contar_primos(10)

    filas = []
    for h in lista_hilos:
        total, tiempo = medir(limite, h)
        print(f"[numba] h={h} primos={total} tiempo={tiempo:.4f}s")
        filas.append(("numba", h, tiempo, total, limite))

    ruta = Path("tiempos_numba.csv")
    with ruta.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["modo", "hilos", "tiempo_seg", "primos", "limite"])
        writer.writerows(filas)

    print(f"datos guardados en {ruta}")


if __name__ == "__main__":
    main()
