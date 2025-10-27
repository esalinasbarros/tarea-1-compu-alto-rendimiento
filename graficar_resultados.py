import csv
from pathlib import Path

import matplotlib.pyplot as plt


def cargar_csv(ruta):
    datos = {}
    if not Path(ruta).exists():
        print(f"no encuentro {ruta}, saltando")
        return datos
    with open(ruta) as f:
        reader = csv.DictReader(f)
        for fila in reader:
            hilos = int(fila["hilos"])
            modo = fila["modo"]
            tiempo = float(fila["tiempo_seg"])
            if hilos not in datos:
                datos[hilos] = {}
            datos[hilos][modo] = tiempo
    return datos


def juntar(*tablas):
    output = {}
    for tabla in tablas:
        for h, modos in tabla.items():
            if h not in output:
                output[h] = {}
            output[h].update(modos)
    return output


def main():
    tabla = juntar(
        cargar_csv("tiempos_openmp.csv"),
        cargar_csv("tiempos_numba.csv"),
    )
    if not tabla:
        print("sin datos para graficar :(")
        return

    labels = sorted({modo for m in tabla.values() for modo in m})
    colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#8c564b"]
    while len(colores) < len(labels):
        colores.extend(colores)

    xs = sorted(tabla.keys())
    fig, ax = plt.subplots()
    for label, color in zip(labels, colores):
        ys = []
        for x in xs:
            ys.append(tabla[x].get(label, None))
        puntos_x = [x for x, y in zip(xs, ys) if y is not None]
        puntos_y = [y for y in ys if y is not None]
        if not puntos_x:
            continue
        ax.plot(puntos_x, puntos_y, marker="o", label=label, color=color[:7])

    ax.set_xlabel("Número de hilos / procesadores")
    ax.set_ylabel("Tiempo (segundos)")
    ax.set_title("Conteo de primos: OpenMP vs Numba")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
