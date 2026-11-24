"""
EXTRACCION DE CARACTERISTICAS + grafico 2D.

    hu1 ... hu4   Momentos de Hu (en escala logaritmica), invariantes a
                  traslacion, escala y rotacion (hu5-hu7 se calculan pero no se
                  usan por defecto: son ruido en formas simetricas)
    circularidad  4*pi*area / perimetro^2   (1 = circulo perfecto)
    vertices      cantidad de vertices del contorno aproximado por un poligono

Hay ademas tres caracteristicas OPCIONALES ya calculadas (no se usan por defecto):
    solidez       area / area de la envolvente convexa
    aspecto       lado largo / lado corto del rectangulo minimo rotado
    agujeros      cantidad de agujeros del objeto (arandelas y tuercas: 1)


Si las imagenes estan en subcarpetas por clase (tuercas/, clavos/, ...), el nombre de
la subcarpeta se toma como clase real; sirve para colorear el grafico y evaluar.

Uso suelto (solo extrae y grafica, sin K-means):
    python caracteristicas.py -i imagenes_binarias -o resultados
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------
# CONFIGURACION
# ----------------------------------------------------------------------
# Se usan Hu 1-4. Los Hu 5-7 valen ~0 (1e-17) en formas simetricas como circulos y
# hexagonos: en logaritmo son ruido numerico con signo aleatorio que arruina K-means.
CARACTERISTICAS = ["hu1", "hu2", "hu3", "hu4", "circularidad", "vertices"]
EPSILON_VERTICES = 0.01      # tolerancia del poligono: fraccion del perimetro
                             # (mas chico = mas vertices; mas grande = menos)
AGUJERO_MIN_FRAC = 0.005     # un agujero cuenta si mide al menos esta fraccion del area
EJES_PLOT = ("circularidad", "vertices")   # par de caracteristicas del grafico 3

TODAS = ["hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7",
         "circularidad", "vertices", "solidez", "aspecto", "agujeros"]


# ----------------------------------------------------------------------
# Extraccion
# ----------------------------------------------------------------------
def extraer(binaria):
    """Devuelve un diccionario {nombre: valor} con todas las caracteristicas."""
    binaria = (binaria > 127).astype(np.uint8) * 255

    # Contornos con jerarquia: los hijos son agujeros
    contornos, jerarquia = cv2.findContours(binaria, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    externos = [i for i, h in enumerate(jerarquia[0]) if h[3] == -1]
    i_ext = max(externos, key=lambda i: cv2.contourArea(contornos[i]))
    c = contornos[i_ext]

    area = cv2.contourArea(c)
    perimetro = cv2.arcLength(c, True)

    # Momentos de Hu sobre la imagen binaria 
    m = cv2.moments(binaria, binaryImage=True)
    hu = cv2.HuMoments(m).ravel()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)  

    circularidad = min(4.0 * np.pi * area / perimetro ** 2, 1.0) if perimetro > 0 else 0.0
    vertices = len(cv2.approxPolyDP(c, EPSILON_VERTICES * perimetro, True))

    hull_area = cv2.contourArea(cv2.convexHull(c))
    solidez = area / hull_area if hull_area > 0 else 0.0

    (_, _), (w, h), _ = cv2.minAreaRect(c)
    aspecto = max(w, h) / max(min(w, h), 1e-6)

    agujeros = sum(1 for i, hj in enumerate(jerarquia[0])
                   if hj[3] == i_ext and cv2.contourArea(contornos[i]) >= AGUJERO_MIN_FRAC * area)

    d = {f"hu{k + 1}": float(hu_log[k]) for k in range(7)}
    d.update(circularidad=float(circularidad), vertices=float(vertices),
             solidez=float(solidez), aspecto=float(aspecto), agujeros=float(agujeros))
    return d


def vector(binaria, nombres=CARACTERISTICAS):
    d = extraer(binaria)
    return np.array([d[n] for n in nombres], dtype=float)


def extraer_carpeta(carpeta, nombres=CARACTERISTICAS):
    """Recorre las imagenes binarias de una carpeta (ignora carpetas que empiezan con '_',
    como _debug). Devuelve X (n, d), lista de rutas relativas y lista de clases reales
    (None si las imagenes no estan en subcarpetas)."""
    carpeta = Path(carpeta)
    archivos = sorted(f for f in carpeta.rglob("*.png")
                      if not any(p.startswith("_") for p in f.relative_to(carpeta).parts))
    X, rutas, clases = [], [], []
    for f in archivos:
        img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None or not (img > 127).any():
            print(f"  [!] Imagen vacia o ilegible, se omite: {f}")
            continue
        X.append(vector(img, nombres))
        rel = f.relative_to(carpeta)
        rutas.append(str(rel))
        clases.append(rel.parts[0] if len(rel.parts) > 1 else None)
    return np.array(X), rutas, clases


def guardar_csv(ruta, X, rutas, clases, nombres):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["imagen", "clase"] + list(nombres))
        for r, c, fila in zip(rutas, clases, X):
            w.writerow([r, c or ""] + [f"{v:.6g}" for v in fila])


# ----------------------------------------------------------------------
# Graficos 2D
# ----------------------------------------------------------------------
def pca_2d(Xz):
    """Proyecta a 2D con PCA (SVD). Devuelve (puntos_2d, media, componentes, var_explicada)."""
    media = Xz.mean(axis=0)
    _, s, vt = np.linalg.svd(Xz - media, full_matrices=False)
    comp = vt[:2]
    var = (s ** 2) / np.sum(s ** 2)
    return (Xz - media) @ comp.T, media, comp, var[:2]


_OFFSETS = [(4, 4), (4, -9), (-4, 4), (-4, -9)]   # reparte las etiquetas para que se pisen menos


def _dispersion(ax, P, grupos, nombres_grupo, titulo, xl, yl, centros=None, jitter_x=False,
                es_val=None, textos=None, resaltar=None, con_nombres=False, colores=None):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    P = P.copy()
    if jitter_x:     # el eje X son enteros (vertices): se separa un poco para ver puntos apilados
        P[:, 0] += rng.uniform(-0.15, 0.15, len(P))
    val = np.zeros(len(P), bool) if es_val is None else np.asarray(es_val, bool)

    for i, g in enumerate(sorted(set(grupos), key=str)):
        m = np.array([x == g for x in grupos])
        col = plt.get_cmap("tab10")((colores or {}).get(g, i) % 10)
        ax.scatter(P[m & ~val, 0], P[m & ~val, 1], s=45, alpha=0.85, color=col, edgecolor="k",
                   linewidth=0.5, label=nombres_grupo.get(g, str(g)))
        if (m & val).any():     # triangulos = imagenes de validacion (no entrenaron)
            ax.scatter(P[m & val, 0], P[m & val, 1], s=65, alpha=0.9, color=col, marker="^",
                       edgecolor="k", linewidth=0.6)
    if val.any():
        ax.scatter([], [], marker="^", c="white", edgecolor="k", label="validacion (triangulos)")
    if resaltar is not None and len(resaltar):
        r = np.asarray(resaltar, int)
        ax.scatter(P[r, 0], P[r, 1], s=280, facecolors="none", edgecolors="k", linewidths=2,
                   label="semillas elegidas", zorder=4)
    if centros is not None:
        ax.scatter(centros[:, 0], centros[:, 1], marker="X", s=220, c="red",
                   edgecolor="k", linewidth=1.2, label="centroides", zorder=5)
    if con_nombres and textos is not None:
        for i, (x, y) in enumerate(P):
            dx, dy = _OFFSETS[i % 4]
            ax.annotate(textos[i], (x, y), xytext=(dx, dy), textcoords="offset points",
                        fontsize=6.5, ha="left" if dx > 0 else "right", alpha=0.9)
    ax.set_title(titulo)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax._datos_hover = (P, textos)


def _activar_hover(fig, axs):
    """En la ventana interactiva (--mostrar), al pasar el mouse sobre un punto aparece
    el nombre de su imagen."""
    anot = {}
    for ax in axs:
        a = ax.annotate("", xy=(0, 0), xytext=(14, 14), textcoords="offset points", zorder=10,
                        bbox=dict(boxstyle="round", fc="w", alpha=0.95),
                        arrowprops=dict(arrowstyle="->"))
        a.set_visible(False)
        anot[ax] = a

    def al_mover(ev):
        for a in anot.values():
            a.set_visible(False)
        ax = ev.inaxes
        datos = getattr(ax, "_datos_hover", None)
        if ax is not None and datos is not None and datos[1] is not None:
            P, textos = datos
            px = ax.transData.transform(P)
            d = np.hypot(px[:, 0] - ev.x, px[:, 1] - ev.y)
            i = int(d.argmin())
            if d[i] < 12:
                anot[ax].xy = P[i]
                anot[ax].set_text(textos[i])
                anot[ax].set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", al_mover)
    return al_mover


def graficar(Xz, X, nombres, clases, ruta_png, etiquetas=None, centroides_z=None,
             nombres_cluster=None, mostrar=False, textos=None, es_val=None,
             resaltar=None, con_nombres=False, clase_cluster=None):
    """Figura con 2 o 3 paneles:
       A) PCA 2D coloreado por clase real (o un solo color si no hay clases)
       B) PCA 2D coloreado por cluster de K-means + centroides (si hay etiquetas)
       C) Dos caracteristicas originales (EJES_PLOT) coloreadas por cluster o por clase

    textos      : nombre de cada imagen (para etiquetas y hover)
    es_val      : mascara booleana; True = imagen de validacion (se dibuja como triangulo)
    resaltar    : indices de imagenes a marcar con un aro (semillas elegidas)
    con_nombres : escribe el nombre junto a cada punto (figura mas grande)
    clase_cluster: {cluster: clase}; si se da, cada clase usa el MISMO color en todos los paneles
    """
    import matplotlib
    if not mostrar:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P, media, comp, var = pca_2d(Xz)
    xl = f"Componente 1 ({var[0] * 100:.0f} % de la varianza)"
    yl = f"Componente 2 ({var[1] * 100:.0f} % de la varianza)"

    hay_clases = all(c is not None for c in clases)
    grupos_real = list(clases) if hay_clases else ["imagenes"] * len(P)
    n_pan = 3 if etiquetas is not None else 2
    fig, axs = plt.subplots(1, n_pan, figsize=((9 * n_pan, 8) if con_nombres else (6 * n_pan, 5.2)))
    extra = dict(es_val=es_val, textos=textos, resaltar=resaltar, con_nombres=con_nombres)

    # Mismo color para la misma clase en todos los paneles (si los nombres de cluster son clases)
    col_real = {cl: i for i, cl in enumerate(sorted(set(clases)))} if hay_clases else None
    col_cl = None
    if etiquetas is not None and hay_clases and clase_cluster:
        vals = [clase_cluster.get(j) for j in sorted(set(int(e) for e in etiquetas))]
        if all(v in col_real for v in vals) and len(set(vals)) == len(vals):
            col_cl = {j: col_real[clase_cluster[j]] for j in sorted(set(int(e) for e in etiquetas))}

    _dispersion(axs[0], P, grupos_real, {}, "PCA 2D - clase real" if hay_clases
                else "PCA 2D - puntos", xl, yl, colores=col_real, **extra)

    if etiquetas is not None:
        C = (centroides_z - media) @ comp.T
        _dispersion(axs[1], P, list(etiquetas), nombres_cluster or {},
                    "PCA 2D - clusters K-means", xl, yl, centros=C, colores=col_cl, **extra)

    ix, iy = nombres.index(EJES_PLOT[0]), nombres.index(EJES_PLOT[1])
    Q = X[:, [ix, iy]]
    _dispersion(axs[-1], Q, list(etiquetas) if etiquetas is not None else grupos_real,
                nombres_cluster or {},
                f"{EJES_PLOT[1]} vs {EJES_PLOT[0]}" + (" - clusters" if etiquetas is not None else ""),
                EJES_PLOT[0], EJES_PLOT[1], jitter_x=(EJES_PLOT[0] == "vertices"),
                colores=col_cl if etiquetas is not None else col_real, **extra)

    fig.tight_layout()
    ruta_png = Path(ruta_png)
    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_png, dpi=100 if con_nombres else 130)
    if mostrar:
        if textos is not None:
            _activar_hover(fig, axs)
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------
# Uso suelto
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Extraccion de caracteristicas + grafico 2D")
    p.add_argument("-i", "--entrada", default="imagenes_binarias")
    p.add_argument("-o", "--salida", default="resultados")
    p.add_argument("--mostrar", action="store_true", help="abre la ventana del grafico")
    a = p.parse_args()

    if not Path(a.entrada).is_dir():
        sys.exit(f"No existe la carpeta: {a.entrada}")
    X, rutas, clases = extraer_carpeta(a.entrada)
    if len(X) == 0:
        sys.exit("No se encontraron imagenes binarias validas.")

    from kmeans import estandarizar
    Xz, _, _ = estandarizar(X)
    guardar_csv(Path(a.salida) / "caracteristicas.csv", X, rutas, clases, CARACTERISTICAS)
    graficar(Xz, X, CARACTERISTICAS, clases, Path(a.salida) / "puntos_2d.png", mostrar=a.mostrar)
    print(f"{len(X)} imagenes -> {a.salida}/caracteristicas.csv y puntos_2d.png")


if __name__ == "__main__":
    main()