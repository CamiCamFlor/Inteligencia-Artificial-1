"""
ENTRENAMIENTO DEL K-NN DE COMANDOS DE VOZ ("proporcion", "contar", "salir").

    1. Toma los audios de la carpeta de originales y los preprocesa     (preprocesar_audio.py)
    2. Extrae las 160 caracteristicas de cada audio                       (caracteristicas_audio.py)
    3. Reserva una parte para VALIDAR (o usa una carpeta de validacion aparte)
    4. Estandariza con la media y el desvio del ENTRENAMIENTO
    5. Elige K por validacion "dejando uno afuera" sobre el entrenamiento (K-NN propio, knn.py)
    6. Informa la exactitud, guarda el modelo y dibuja el PCA 2D

Los audios se organizan en subcarpetas, una por comando:
    audios_originales/proporcion/..., /contar/..., /salir/...

Uso:
    python entrenar_audio.py
    python entrenar_audio.py -k 5 --pesos distancia
    python entrenar_audio.py -i entrenamiento --carpeta-validacion validacion
    python entrenar_audio.py --saltear-preproceso --mostrar       (ventana con nombre al pasar el mouse)
    python entrenar_audio.py --umbral-db 30 --silencios todos      (opciones de preprocesar_audio.py)

Si se pasa --validacion 0 se entrena con TODO (usarlo al final, ya decidido K, para el modelo definitivo).
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import caracteristicas_audio as ca
import knn
import preprocesar_audio
from knn import KNN, estandarizar

# ----------------------------------------------------------------------
# CONFIGURACION POR DEFECTO
# ----------------------------------------------------------------------
CARPETA_ORIGINALES = "Audios_Entrenamiento"
CARPETA_PROCESADOS = "audios_procesados"
CARPETA_RESULTADOS = "resultados_audio"
FRACCION_VALIDACION = 0.3   # porcentaje de cada comando que NO se usa para entrenar


# ----------------------------------------------------------------------
# Auxiliares
# ----------------------------------------------------------------------
def dividir(clases, fraccion, semilla):
    """Mascara es_val, estratificada por comando; siempre quedan >= 2 audios de cada uno en entrenamiento."""
    es_val = np.zeros(len(clases), bool)
    if fraccion <= 0:
        return es_val
    rng = np.random.default_rng(semilla)
    for c in sorted(set(clases)):
        idx = [i for i, cl in enumerate(clases) if cl == c]
        rng.shuffle(idx)
        es_val[idx[:min(int(round(len(idx) * fraccion)), max(len(idx) - 2, 0))]] = True
    return es_val


def tabla_confusion(titulo, reales, pred, clases, nombres=None):
    """Imprime clase real x clase predicha. Devuelve la exactitud (0-1)."""
    print(f"\n{titulo}  ({len(reales)} audios)")
    ancho = max(len(c) for c in list(clases) + ["real \\ predicha"]) + 2
    print("real \\ predicha".ljust(ancho) + "".join(c[:10].rjust(12) for c in clases))
    for c in clases:
        fila = [sum(1 for r, p in zip(reales, pred) if r == c and p == q) for q in clases]
        print(c.ljust(ancho) + "".join(str(v).rjust(12) for v in fila))
    aciertos = sum(1 for r, p in zip(reales, pred) if r == p)
    print(f"Exactitud: {aciertos / len(reales) * 100:.1f} %  ({aciertos}/{len(reales)})")
    if nombres is not None:
        for n, r, p in zip(nombres, reales, pred):
            if r != p:
                print(f"   error: {n}  (real: {r}, predicho: {p})")
    return aciertos / len(reales)


def pca_2d(Z):
    """Proyeccion a 2D con PCA (SVD). Devuelve (puntos 2D, varianza explicada de las 2 componentes)."""
    Zc = Z - Z.mean(axis=0)
    _, s, vt = np.linalg.svd(Zc, full_matrices=False)
    return Zc @ vt[:2].T, (s ** 2 / np.sum(s ** 2))[:2]


# ----------------------------------------------------------------------
# Graficos
# ----------------------------------------------------------------------
_OFFSETS = [(4, 4), (4, -9), (-4, 4), (-4, -9)]


def graficar_pca(Z, clases, es_val, textos, ruta_png, mostrar=False, con_nombres=False):
    """PCA 2D de las caracteristicas estandarizadas, coloreado por comando.
    Circulos = entrenamiento, triangulos = validacion, X = centro de cada comando,
    elipse punteada = region de +-2 desvios de cada comando."""
    import matplotlib
    if not mostrar:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    P, var = pca_2d(Z)
    clases = np.asarray(clases)
    nombres = sorted(set(clases.tolist()))
    fig, ax = plt.subplots(figsize=(13, 10) if con_nombres else (8.5, 6.8))

    for i, c in enumerate(nombres):
        m = clases == c
        col = plt.get_cmap("tab10")(i)
        ax.scatter(P[m & ~es_val, 0], P[m & ~es_val, 1], s=50, color=col, edgecolor="k",
                   linewidth=0.5, alpha=0.85, label=c)
        if (m & es_val).any():
            ax.scatter(P[m & es_val, 0], P[m & es_val, 1], s=75, color=col, marker="^",
                       edgecolor="k", linewidth=0.7, alpha=0.95)
        pts = P[m]
        ax.scatter(*pts.mean(axis=0), marker="X", s=230, color=col, edgecolor="k", linewidth=1.5, zorder=5)
        if len(pts) >= 3:
            val, vec = np.linalg.eigh(np.cov(pts.T))
            val = np.maximum(val, 1e-12)
            ax.add_patch(Ellipse(pts.mean(axis=0), 4 * np.sqrt(val[1]), 4 * np.sqrt(val[0]),
                                 angle=np.degrees(np.arctan2(vec[1, 1], vec[0, 1])), fill=False,
                                 edgecolor=col, linewidth=1.6, linestyle="--", alpha=0.85))
    if es_val.any():
        ax.scatter([], [], marker="^", c="white", edgecolor="k", label="validacion (triangulos)")
    ax.scatter([], [], marker="X", c="white", edgecolor="k", s=100, label="centro del comando")

    if con_nombres:
        for i, (x, y) in enumerate(P):
            dx, dy = _OFFSETS[i % 4]
            ax.annotate(textos[i], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.5,
                        ha="left" if dx > 0 else "right", alpha=0.9)

    ax.set_title(f"PCA 2D de las {Z.shape[1]} caracteristicas  "
                 f"(las 2 componentes explican {var.sum() * 100:.0f} % de la varianza)")
    ax.set_xlabel(f"Componente 1 ({var[0] * 100:.0f} %)")
    ax.set_ylabel(f"Componente 2 ({var[1] * 100:.0f} %)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_png, dpi=100 if con_nombres else 130)

    if mostrar and textos is not None:        # al pasar el mouse sobre un punto aparece su nombre
        anot = ax.annotate("", xy=(0, 0), xytext=(14, 14), textcoords="offset points", zorder=10,
                           bbox=dict(boxstyle="round", fc="w", alpha=0.95), arrowprops=dict(arrowstyle="->"))
        anot.set_visible(False)

        def al_mover(ev):
            anot.set_visible(False)
            if ev.inaxes is ax:
                px = ax.transData.transform(P)
                d = np.hypot(px[:, 0] - ev.x, px[:, 1] - ev.y)
                i = int(d.argmin())
                if d[i] < 12:
                    anot.xy = P[i]
                    anot.set_text(textos[i])
                    anot.set_visible(True)
            fig.canvas.draw_idle()
        fig.canvas.mpl_connect("motion_notify_event", al_mover)
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Entrena el K-NN de comandos de voz",
        epilog="Tambien acepta las opciones de preprocesar_audio.py (--umbral-db, --silencios, --debug...).")
    p.add_argument("-i", "--originales", default=CARPETA_ORIGINALES)
    p.add_argument("-b", "--procesados", default=CARPETA_PROCESADOS)
    p.add_argument("-r", "--resultados", default=CARPETA_RESULTADOS)
    p.add_argument("-k", "--vecinos", type=int, default=3,
               help="cantidad de vecinos del K-NN (por defecto: 3)")
    p.add_argument("--metrica", choices=["euclidea", "manhattan"], default="euclidea")
    p.add_argument("--pesos", choices=["uniforme", "distancia"], default="uniforme")
    p.add_argument("--validacion", type=float, default=FRACCION_VALIDACION,
                   help="fraccion de cada comando reservada para validar (0 = entrenar con todo)")
    p.add_argument("--carpeta-validacion", metavar="CARPETA",
                   help="audios ORIGINALES de validacion ya separados (reemplaza al sorteo)")
    p.add_argument("--azar", type=int, default=0, help="semilla del sorteo entrenamiento/validacion")
    p.add_argument("--mostrar", action="store_true", help="abre el grafico PCA con nombres al pasar el mouse")
    p.add_argument("--saltear-preproceso", action="store_true",
                   help="usa los audios que ya estan procesados en la carpeta de procesados")
    args, resto = p.parse_known_args()
    proc_val = str(Path(args.procesados).with_name(Path(args.procesados).name + "_validacion"))

    # 1) Preprocesamiento
    if not args.saltear_preproceso:
        print(f"[1/5] Preprocesando  {args.originales}  ->  {args.procesados}")
        preprocesar_audio.procesar_carpeta(preprocesar_audio.construir_parser().parse_args(
            ["-i", args.originales, "-o", args.procesados] + resto))
        if args.carpeta_validacion:
            print(f"      Validacion: {args.carpeta_validacion}  ->  {proc_val}")
            preprocesar_audio.procesar_carpeta(preprocesar_audio.construir_parser().parse_args(
                ["-i", args.carpeta_validacion, "-o", proc_val] + resto))
    elif resto:
        print(f"[!] Argumentos ignorados al saltear el preproceso: {resto}")

    # 2) Caracteristicas
    print(f"\n[2/5] Extrayendo {len(ca.nombres_caracteristicas())} caracteristicas por audio")
    X, rutas, clases = ca.extraer_carpeta(args.procesados)
    if len(X) == 0:
        sys.exit("No hay audios procesados para analizar.")
    val_fija = None
    if args.carpeta_validacion:
        Xv, rv, cv = ca.extraer_carpeta(proc_val)
        if len(Xv) == 0:
            sys.exit(f"No hay audios procesados de validacion en {proc_val}.")
        val_fija = np.r_[np.zeros(len(X), bool), np.ones(len(Xv), bool)]
        X, rutas, clases = np.vstack([X, Xv]), rutas + [f"validacion/{r}" for r in rv], clases + cv
    if not all(c is not None for c in clases):
        sys.exit("Los audios tienen que estar en subcarpetas, una por comando "
                 "(por ejemplo audios_originales/contar/...).")
    y = np.array(clases)
    comandos = sorted(set(clases))
    print(f"      {len(X)} audios, comandos: " + ", ".join(f"{c} ({n})" for c, n in sorted(Counter(clases).items())))

    # 3) Entrenamiento / validacion y estandarizacion (solo con estadisticas del entrenamiento)
    es_val = val_fija if val_fija is not None else dividir(clases, args.validacion, args.azar)
    tr = ~es_val
    cuentas = Counter(y[tr])
    if len(cuentas) < 2 or min(cuentas.values()) < 2:
        sys.exit("Hacen falta al menos 2 audios de entrenamiento por comando (y 2 comandos).")
    print(f"\n[3/5] Entrenamiento: {int(tr.sum())} audios  |  validacion: {int(es_val.sum())}")
    _, media, desvio = estandarizar(X[tr])
    Z = estandarizar(X, media, desvio)[0]

    # 4) Eleccion de K por leave-one-out sobre el entrenamiento
    k = args.vecinos
    if k < 1 or k > int(tr.sum()):
        sys.exit(f"K debe estar entre 1 y {int(tr.sum())}.")
    print(f"\n[4/5] K-NN ({args.metrica}, pesos {args.pesos}, K={k})")
    knn = KNN(k, args.metrica, args.pesos).fit(Z[tr], y[tr])

    
    # 5) Resultados
    print("\n[5/5] Resultados")
    pred_val = knn.predict(Z[es_val]) if es_val.any() else np.array([])
    if es_val.any():
        tabla_confusion(
            "VALIDACION (audios que NO se usaron para entrenar)",
            list(y[es_val]),
            list(pred_val),
            comandos,
            [r for r, v in zip(rutas, es_val) if v]
        )
    else:
        print("\nNo hay conjunto de validacion independiente.")
        if es_val.any():
            tabla_confusion("VALIDACION (audios que NO se usaron para entrenar)", list(y[es_val]),
                            list(pred_val), comandos, [r for r, v in zip(rutas, es_val) if v])
        else:
            print("\n(sin validacion: se entreno con todos los audios; la exactitud confiable es la de leave-one-out)")

    res = Path(args.resultados)
    res.mkdir(parents=True, exist_ok=True)
    ca.guardar_csv(res / "caracteristicas_audio.csv", X, rutas, clases)
    pred_todos = np.empty(len(X), dtype=object)
    pred_todos[tr] = knn.predict(Z[tr])
    pred_todos[es_val] = pred_val
    with open(res / "predicciones_audio.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["audio", "comando_real", "comando_predicho", "conjunto", "como_se_predijo"])
        for i, r in enumerate(rutas):
            w.writerow([r, y[i], pred_todos[i], "validacion" if es_val[i] else "entrenamiento",
                        "modelo"])

    pj = Path(args.procesados) / "_preproceso_audio.json"
    knn.guardar(res / "modelo_knn_audio.npz", media=media, desvio=desvio,
                caracteristicas=np.array(ca.nombres_caracteristicas()),
                parametros_preproceso=np.array(pj.read_text(encoding="utf-8") if pj.exists() else ""),
                parametros_caracteristicas=np.array(json.dumps(dict(
                    N_SEGMENTOS=ca.N_SEGMENTOS, MFCC_INDICES=ca.MFCC_INDICES, N_MFCC=ca.N_MFCC,
                    N_MELS=ca.N_MELS, FMIN=ca.FMIN, FMAX=ca.FMAX,
                    VENTANA_MS=ca.VENTANA_MS, SALTO_MS=ca.SALTO_MS))))

    textos = [Path(r).stem for r in rutas]
    graficar_pca(Z, y, es_val, textos, res / "pca_audio.png")
    graficar_pca(Z, y, es_val, textos, res / "pca_audio_nombres.png", mostrar=args.mostrar, con_nombres=True)
    
    _, var = pca_2d(Z)
    print(f"\nPCA: las 2 componentes explican el {var.sum() * 100:.0f} % de la varianza "
          "(es solo una proyeccion: el K-NN trabaja con las caracteristicas completas)")
    print(f"Guardado en {res}/: modelo_knn_audio.npz, predicciones_audio.csv, "
      "caracteristicas_audio.csv, pca_audio.png, pca_audio_nombres.png")


if __name__ == "__main__":
    main()
