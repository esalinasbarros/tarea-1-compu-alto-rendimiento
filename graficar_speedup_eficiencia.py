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


def series_por_modo(tabla, pred):
    modos = {}
    for h, d in tabla.items():
        for modo, t in d.items():
            if pred(modo):
                modos.setdefault(modo, []).append((h, t))
    modos = {m: sorted(vals, key=lambda x: x[0]) for m, vals in modos.items()}
    return modos


def calcular_SE(series, T1):
    speedup, eficiencia = {}, {}
    for modo, pares in series.items():
        s, e = {}, {}
        for h, t in pares:
            if h <= 0 or t <= 0:
                continue
            S = T1 / t
            E = S / h
            s[h] = S
            e[h] = E
        if s:
            speedup[modo] = s
            eficiencia[modo] = e
    return speedup, eficiencia


def plot_curvas(curvas, titulo, ylabel, salida_png):
    if not curvas:
        print(f"no hay datos para {titulo}")
        return
    plt.figure()
    for label, serie in sorted(curvas.items()):
        xs = sorted(serie.keys())
        ys = [serie[h] for h in xs]
        plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("Número de procesadores / hilos")
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(salida_png, dpi=160)
    print(f"guardado: {salida_png}")


def main():
    tabla = juntar(
        cargar_csv("tiempos_openmp.csv"),
        cargar_csv("tiempos_numba.csv"),
    )
    if not tabla:
        print("sin datos para graficar :(")
        return

    T1_omp = None
    if 1 in tabla and "secuencial" in tabla[1]:
        T1_omp = float(tabla[1]["secuencial"])
    else:
        print("no se encontro baseline T1 para omp")

    T1_numba = None
    if 1 in tabla and "numba" in tabla[1]:
        T1_numba = float(tabla[1]["numba"])
    else:
        print("no se encontro baseline T1 para numba")

    omp_series = series_por_modo(tabla, pred=lambda m: m.startswith("omp_") and m != "secuencial")
    numba_series = series_por_modo(tabla, pred=lambda m: m == "numba")

    curvas_S, curvas_E = {}, {}

    if T1_omp is not None and omp_series:
        S_omp, E_omp = calcular_SE(omp_series, T1_omp)
        curvas_S.update({f"OpenMP {m}": s for m, s in S_omp.items()})
        curvas_E.update({f"OpenMP {m}": e for m, e in E_omp.items()})

    if T1_numba is not None and numba_series:
        S_num, E_num = calcular_SE(numba_series, T1_numba)
        if "numba" in S_num:
            curvas_S["Numba"] = S_num["numba"]
            curvas_E["Numba"] = E_num["numba"]

    plot_curvas(curvas_S, "Speedup vs número de procesadores (OpenMP y Numba)",
                "Speedup S(p) = T(1) / T(p)", "speedup_vs_hilos.png")
    plot_curvas(curvas_E, "Eficiencia vs número de procesadores (OpenMP y Numba)",
                "Eficiencia E(p) = S(p) / p", "eficiencia_vs_hilos.png")


if __name__ == "__main__":
    main()
