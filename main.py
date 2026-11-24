"""
MAIN: pipeline completo de clasificacion de piezas (tuercas, tornillos, clavos, arandelas).

    1. Carga las imagenes de una carpeta y las preprocesa      (preprocesar.py)
    2. Extrae sus caracteristicas                               (caracteristicas.py)
    3. Define los centroides iniciales a partir de ejemplos ETIQUETADOS
    4. Entrena K-means con parte de las imagenes
    5. Valida con las imagenes que NO se usaron para entrenar
    6. Informa cuantas piezas de cada clase hay y guarda tablas y graficos

Las imagenes etiquetadas se indican con subcarpetas por clase:
    imagenes_originales/tuercas/..., /tornillos/..., /clavos/..., /arandelas/...

COMO SE INICIALIZAN LOS CENTROIDES (--init)
    auto      (por defecto) "clases" si las imagenes estan en subcarpetas; si no, k-means++
    clases    cada centroide arranca en el promedio de las imagenes de entrenamiento de
              esa clase
    manual    cada centroide arranca en UNA imagen que vos elegis (--semillas)
    kmeans++  sin etiquetas: puntos iniciales automaticos (los clusters se nombran por mayoria)
    random    sin etiquetas: puntos iniciales al azar

Luego K-means refina los centroides con las imagenes de entrenamiento
(--no-refinar los deja fijos, tal como se inicializaron).

Uso:
    python main.py
    python main.py --validacion 0.3 --azar 1
    python main.py -i entrenamiento --carpeta-validacion validacion      (carpetas separadas)
    python main.py --saltear-preproceso --init manual \\
        --semillas tuercas_03 tornillos_05 clavos_01 arandelas_07
    python main.py --umbral relativo --factor-relativo 0.8       (opciones de preprocesar.py)
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import caracteristicas as car
import preprocesar
from kmeans import KMeans, estandarizar

# ----------------------------------------------------------------------
# CONFIGURACION POR DEFECTO
# ----------------------------------------------------------------------
CARPETA_ORIGINALES = "G:\Mi unidad\Inteligencia Artificial 1"
CARPETA_BINARIAS = "imagenes_binarias"
CARPETA_RESULTADOS = "resultados"
K = 4
FRACCION_VALIDACION = 0.3   


def resolver_semillas(pedidas, rutas):
    """Convierte nombres de imagen (con o sin carpeta/extension) en indices."""
    idx = []
    for s in pedidas:
        s_norm = Path(s).with_suffix("").as_posix()
        cand = [i for i, r in enumerate(rutas)
                if Path(r).with_suffix("").as_posix() == s_norm or Path(r).stem == s_norm]
        if len(cand) != 1:
            motivo = "no encontre ninguna imagen" if not cand else \
                f"hay {len(cand)} imagenes con ese nombre (usa carpeta/nombre)"
            sys.exit(f"Semilla '{s}': {motivo}.")
        idx.append(cand[0])
    return idx


def dividir(clases, fijas, fraccion, semilla):
    """Separa validacion de entrenamiento, por clase (estratificado). Las imagenes en
    'fijas' (semillas) siempre quedan en entrenamiento. Devuelve mascara es_val."""
    es_val = np.zeros(len(clases), bool)
    if fraccion <= 0 or not all(c is not None for c in clases):
        return es_val
    rng = np.random.default_rng(semilla)
    for c in sorted(set(clases)):
        idx = [i for i, cl in enumerate(clases) if cl == c and i not in fijas]
        rng.shuffle(idx)
        n_val = min(int(round(len(idx) * fraccion)), max(len(idx) - 1, 0))
        es_val[idx[:n_val]] = True
    return es_val


def nombrar_por_mayoria(etiquetas, clases, k):
    """Sin centroides etiquetados: cada cluster toma la clase mayoritaria de sus miembros.
    Devuelve (nombre_para_mostrar, clase_mayoritaria) como dos diccionarios."""
    if not all(c is not None for c in clases):
        return {j: f"Cluster {j}" for j in range(k)}, {j: None for j in range(k)}
    nombres, mayoria = {}, {}
    for j in range(k):
        miembros = [c for e, c in zip(etiquetas, clases) if e == j]
        mayoria[j] = Counter(miembros).most_common(1)[0][0] if miembros else None
        nombres[j] = f"Cluster {j} (~{mayoria[j]})" if miembros else f"Cluster {j} (vacio)"
    return nombres, mayoria


def tabla_confusion(titulo, reales, pred, nombres_cl, clase_cl, k):
    """Imprime clase real x cluster y devuelve la exactitud (0-1)."""
    print(f"\n{titulo}  ({len(reales)} imagenes)")
    cls = sorted(set(reales))
    ancho = max(len(c) for c in cls + ["clase real"]) + 2
    print("clase real".ljust(ancho) + "".join(f"C{j}".rjust(6) for j in range(k)))
    for c in cls:
        fila = [sum(1 for r, p in zip(reales, pred) if r == c and p == j) for j in range(k)]
        print(c.ljust(ancho) + "".join(str(v).rjust(6) for v in fila))
    print("  " + " | ".join(f"C{j} = {nombres_cl[j]}" for j in range(k)))
    aciertos = sum(1 for r, p in zip(reales, pred) if clase_cl[p] == r)
    exactitud = aciertos / len(reales)
    print(f"Exactitud: {exactitud * 100:.1f} %  ({aciertos}/{len(reales)})")
    return exactitud


def main():
    p = argparse.ArgumentParser(
        description="Pipeline: preprocesado -> caracteristicas -> K-means con centroides etiquetados",
        epilog="Tambien acepta las opciones de preprocesar.py (--umbral, --suavizado, --debug...).")
    p.add_argument("-i", "--originales", default=CARPETA_ORIGINALES)
    p.add_argument("-b", "--binarias", default=CARPETA_BINARIAS)
    p.add_argument("-r", "--resultados", default=CARPETA_RESULTADOS)
    p.add_argument("-k", "--clusters", type=int, default=K)
    p.add_argument("--init", choices=["auto", "clases", "manual", "kmeans++", "random"],
                   default="auto", help="como se inicializan los centroides (ver encabezado)")
    p.add_argument("--semillas", nargs="+", metavar="IMAGEN",
                   help="una imagen por clase para usar como centroide inicial (implica --init manual)")
    p.add_argument("--validacion", type=float, default=FRACCION_VALIDACION,
                   help="fraccion de cada clase reservada para validar (0 = usar todas para entrenar)")
    p.add_argument("--carpeta-validacion", metavar="CARPETA",
                   help="imagenes ORIGINALES de validacion, ya separadas (subcarpetas por clase). "
                        "Si se da, reemplaza al sorteo automatico de --validacion")
    p.add_argument("--azar", type=int, default=0, help="semilla del sorteo entrenamiento/validacion")
    p.add_argument("--no-refinar", action="store_true",
                   help="deja los centroides tal como se inicializaron, sin iterar K-means")
    p.add_argument("--mostrar", action="store_true",
                   help="abre la ventana del grafico (con el nombre de la imagen al pasar el mouse)")
    p.add_argument("--saltear-preproceso", action="store_true",
                   help="no vuelve a preprocesar; usa las binarias que ya estan en la carpeta")
    args, resto = p.parse_known_args()
    k = args.clusters
    bin_val = str(Path(args.binarias).with_name(Path(args.binarias).name + "_validacion"))

    # 1) Carga + preprocesamiento
    if not args.saltear_preproceso:
        print(f"[1/5] Preprocesando  {args.originales}  ->  {args.binarias}")
        pre = preprocesar.construir_parser().parse_args(
            ["-i", args.originales, "-o", args.binarias] + resto)
        preprocesar.procesar_carpeta(pre)
        if args.carpeta_validacion:      # mismos parametros que el entrenamiento
            print(f"      Validacion: {args.carpeta_validacion}  ->  {bin_val}")
            preprocesar.procesar_carpeta(preprocesar.construir_parser().parse_args(
                ["-i", args.carpeta_validacion, "-o", bin_val] + resto))
    elif resto:
        print(f"[!] Argumentos ignorados al saltear el preproceso: {resto}")

    # 2) Caracteristicas
    print(f"\n[2/5] Extrayendo caracteristicas: {', '.join(car.CARACTERISTICAS)}")
    X, rutas, clases = car.extraer_carpeta(args.binarias)
    if len(X) == 0:
        sys.exit("No hay imagenes binarias para analizar.")
    val_fija = None
    if args.carpeta_validacion:
        Xv, rv, cv = car.extraer_carpeta(bin_val)
        if len(Xv) == 0:
            sys.exit(f"No hay imagenes binarias de validacion en {bin_val}.")
        val_fija = np.r_[np.zeros(len(X), bool), np.ones(len(Xv), bool)]
        X = np.vstack([X, Xv])
        rutas = rutas + [f"validacion/{r}" for r in rv]
        clases = clases + cv
    if len(X) < k:
        sys.exit(f"Hay {len(X)} imagenes y se piden {k} clusters.")
    etiquetadas = all(c is not None for c in clases)
    print(f"      {len(X)} imagenes, {X.shape[1]} caracteristicas por imagen"
          + ("" if etiquetadas else "  (sin subcarpetas por clase: no hay etiquetas)"))

    # 3) Modo de inicializacion y particion entrenamiento / validacion
    modo = "manual" if args.semillas else args.init
    if modo == "auto":
        modo = "clases" if etiquetadas else "kmeans++"
    if modo == "clases" and not etiquetadas:
        sys.exit("--init clases necesita las imagenes en subcarpetas por clase.")

    semillas = resolver_semillas(args.semillas, rutas) if modo == "manual" else []
    if modo == "manual":
        if len(semillas) != k:
            sys.exit(f"Hay que indicar {k} semillas (una por cluster) y se dieron {len(semillas)}.")
    es_val = val_fija if val_fija is not None else dividir(clases, set(semillas), args.validacion, args.azar)
    if semillas and es_val[semillas].any():
        sys.exit("Las semillas tienen que ser imagenes de ENTRENAMIENTO, no de la carpeta de validacion.")
    tr = ~es_val
    print(f"\n[3/5] Inicializacion de centroides: {modo}"
          f"   |   entrenamiento: {tr.sum()} imagenes, validacion: {es_val.sum()}")

    # Estandarizacion con estadisticas SOLO del entrenamiento (evita "espiar" la validacion)
    _, media, desvio = estandarizar(X[tr])
    Xz = estandarizar(X, media, desvio)[0]

    nombres_cl = clase_cl = None
    if modo == "clases":
        nombres_clase = sorted(set(clases))
        if len(nombres_clase) != k:
            sys.exit(f"Se piden {k} clusters pero hay {len(nombres_clase)} clases "
                     f"({', '.join(nombres_clase)}). Ajusta -k.")
        init = np.array([Xz[tr & np.array([c == n for c in clases])].mean(axis=0)
                         for n in nombres_clase])
        nombres_cl = clase_cl = dict(enumerate(nombres_clase))
    elif modo == "manual":
        init = Xz[semillas].copy()
        nombres_cl = clase_cl = {j: (clases[i] or Path(rutas[i]).stem) for j, i in enumerate(semillas)}
        print("      semillas: " + ", ".join(f"{nombres_cl[j]} <- {Path(rutas[i]).stem}"
                                             for j, i in enumerate(semillas)))
    else:
        init = "k-means++" if modo == "kmeans++" else modo

    # 4) K-means (parte desde los centroides etiquetados; refina con el entrenamiento)
    print(f"\n[4/5] K-means con K = {k}" + ("  (sin refinar)" if args.no_refinar else ""))
    km = KMeans(n_clusters=k, init=init, n_init=10, random_state=args.azar)
    if args.no_refinar and not isinstance(init, str):
        km.cluster_centers_ = init.copy()
        km.labels_ = km.predict(Xz[tr])
    else:
        km.fit(Xz[tr])
        print(f"      inercia = {km.inertia_:.2f}, iteraciones = {km.n_iter_}")
    pred = km.predict(Xz)                      # todas las imagenes (entrenamiento y validacion)
    if nombres_cl is None:
        nombres_cl, clase_cl = nombrar_por_mayoria(pred, clases, k)

    # 5) Informe
    print("\n[5/5] Resultados")
    if etiquetadas:
        reales = list(clases)
        tabla_confusion("ENTRENAMIENTO", [c for c, v in zip(reales, es_val) if not v],
                        [q for q, v in zip(pred, es_val) if not v], nombres_cl, clase_cl, k)
        if es_val.any():
            tabla_confusion("VALIDACION (imagenes que NO se usaron para entrenar)",
                            [c for c, v in zip(reales, es_val) if v],
                            [q for q, v in zip(pred, es_val) if v], nombres_cl, clase_cl, k)

    print("\nPiezas identificadas por clase (todas las imagenes):")
    for j in range(k):
        reales_j = sum(1 for c in clases if c is not None and c == nombres_cl[j]) if etiquetadas else None
        extra = f"   (reales: {reales_j})" if reales_j is not None else ""
        print(f"  {nombres_cl[j]:<28} {int(np.sum(pred == j)):>4}{extra}")

    dist = np.linalg.norm(Xz - km.cluster_centers_[pred], axis=1)
    print("\nImagen mas cercana al centroide de cada cluster (buena candidata a semilla):")
    for j in range(k):
        miembros = np.where(pred == j)[0]
        if len(miembros):
            i = miembros[np.argmin(dist[miembros])]
            print(f"  {nombres_cl[j]:<28} {Path(rutas[i]).stem}   (distancia {dist[i]:.2f})")

    # Archivos de salida
    res = Path(args.resultados)
    res.mkdir(parents=True, exist_ok=True)
    car.guardar_csv(res / "caracteristicas.csv", X, rutas, clases, car.CARACTERISTICAS)
    with open(res / "asignaciones.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["imagen", "clase_real", "clase_predicha", "cluster", "conjunto",
                    "distancia_al_centroide"])
        for i, r in enumerate(rutas):
            w.writerow([r, clases[i] or "", nombres_cl[int(pred[i])], int(pred[i]),
                        "validacion" if es_val[i] else "entrenamiento", f"{dist[i]:.4f}"])
    pj = Path(args.binarias) / "_preproceso.json"
    params_pre = pj.read_text(encoding="utf-8") if pj.exists() else ""
    np.savez(res / "modelo_kmeans.npz", centroides=km.cluster_centers_, media=media, desvio=desvio,
             caracteristicas=np.array(car.CARACTERISTICAS),
             nombres_clases=np.array([nombres_cl[j] for j in range(k)]),
             parametros_preproceso=np.array(params_pre),
             epsilon_vertices=np.array(car.EPSILON_VERTICES))

    textos = [Path(r).stem for r in rutas]
    comunes = dict(etiquetas=pred, centroides_z=km.cluster_centers_, nombres_cluster=nombres_cl,
                   es_val=es_val if es_val.any() else None, textos=textos,
                   resaltar=semillas or None, clase_cluster=clase_cl)
    car.graficar(Xz, X, car.CARACTERISTICAS, clases, res / "clusters_2d.png", **comunes)
    car.graficar(Xz, X, car.CARACTERISTICAS, clases, res / "clusters_2d_nombres.png",
                 con_nombres=True, mostrar=args.mostrar, **comunes)
    print(f"\nGuardado en {res}/: asignaciones.csv, caracteristicas.csv, modelo_kmeans.npz, "
          "clusters_2d.png, clusters_2d_nombres.png")


if __name__ == "__main__":
    main()