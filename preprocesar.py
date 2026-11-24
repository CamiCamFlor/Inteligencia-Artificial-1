"""
PREPROCESAMIENTO de imagenes (tuercas, tornillos, clavos, arandelas).

Lee las imagenes (incluyendo subcarpetas) y guarda en otra carpeta.    
Respeta la estructura de subcarpetas

Pipeline:
  1. Lectura en escala de grises    
  2. Reduccion de tamano (solo si es muy grande, conservando proporciones)
  3. Suavizado (gaussiano / mediana / bilateral)
  4. Umbralizado (Otsu o adaptativo) con deteccion automatica de polaridad
  5. Morfologia: apertura (saca motas) + cierre (tapa grietas)
  6. Se queda solo con el objeto mas grande (descarta ruido/sombras sueltas)
 
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np


# CONFIGURACION POR DEFECTO 

CARPETA_ENTRADA = "G:\Mi unidad\Inteligencia Artificial 1"
CARPETA_SALIDA = "imagenes_binarias"

TAM_MAX = 512             # lado mayor maximo en pixeles (0 = no redimensionar)
SUAVIZADO = "gaussiano"   # "gaussiano" | "mediana" | "bilateral" | "ninguno"
KERNEL_SUAVIZADO = 5      # impar
UMBRAL = "otsu"           # "otsu" | "adaptativo" | "relativo"
BLOQUE_ADAPTATIVO = 51    # impar; solo para umbral adaptativo
C_ADAPTATIVO = 5          # solo para umbral adaptativo
FACTOR_RELATIVO = 0.85    # solo "relativo": objeto = pixeles mas oscuros que
                          # FACTOR * brillo_del_fondo (0.7 = mas estricto, 0.95 = mas laxo)
DEBUG = False             # guarda un montaje original | binaria en salida/_debug
KERNEL_MORFOLOGIA = 3     # impar
AREA_MINIMA = 0.005       # fraccion minima de la imagen para aceptar el objeto
RECORTAR = False          # True: recorta al objeto (pierde el tamano absoluto)
MARGEN_RECORTE = 10       # pixeles de margen si RECORTAR=True

EXTENSIONES = {".jpg", ".jpeg", ".png"}

# Funciones

def leer_gris(ruta):
    """Lee en escala de grises. Usa imdecode para soportar tildes/espacios en la ruta."""
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_GRAYSCALE)


def guardar_png(ruta, imagen):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", imagen)
    if not ok:
        raise IOError("No se pudo codificar el PNG")
    buf.tofile(str(ruta))


def _impar(k):
    return k if k % 2 == 1 else k + 1

def suavizar(gris, metodo, k):
    if metodo == "gaussiano":
        return cv2.GaussianBlur(gris, (k, k), 0)
    if metodo == "mediana":
        return cv2.medianBlur(gris, k)
    if metodo == "bilateral":
        # Suaviza el ruido pero conserva los bordes
        return cv2.bilateralFilter(gris, 9, 50, 50)
    return gris


def nivel_fondo(gris):
    """Brillo tipico del fondo: mediana de una franja del 3 % en el borde de la imagen."""
    f = max(2, int(0.03 * min(gris.shape)))
    borde = np.concatenate([gris[:f, :].ravel(), gris[-f:, :].ravel(),
                            gris[:, :f].ravel(), gris[:, -f:].ravel()])
    return float(np.median(borde))


def binarizar(gris_suave, metodo, bloque, c, factor=FACTOR_RELATIVO):
    if metodo == "relativo":
        # Umbral atado al brillo REAL del fondo en cada foto: sirve igual para
        # fotos mas claras u oscuras (con o sin flash) y capta piezas metalicas
        # cuyo brillo es parecido al del papel.
        g = gris_suave
        bg = nivel_fondo(g)
        if bg < 128:                      # fondo oscuro: se trabaja en negativo
            g, bg = 255 - g, 255 - bg
        return np.where(g < factor * bg, 255, 0).astype(np.uint8)
    if metodo == "adaptativo":
        return cv2.adaptiveThreshold(gris_suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, _impar(bloque), c)
    _, b = cv2.threshold(gris_suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def objeto_en_blanco(binaria):
    """Si el borde de la imagen es mayoritariamente blanco, el fondo quedo blanco: invierte."""
    borde = np.concatenate([binaria[0, :], binaria[-1, :], binaria[:, 0], binaria[:, -1]])
    return 255 - binaria if borde.mean() > 127 else binaria


def quedarse_con_mayor_objeto(binaria, area_minima_frac):
    """Conserva solo la componente blanca mas grande. Los agujeros (tuercas,
    arandelas) NO se rellenan: son parte de la forma y sirven para clasificar."""
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    mejor = 1 + int(np.argmax(areas))
    if areas.max() < area_minima_frac * binaria.size:
        return None
    return np.where(etiquetas == mejor, 255, 0).astype(np.uint8)


def preprocesar(gris, a):
    # Reduccion de tamano (conserva la proporcion para no deformar la forma)
    if a.tam_max and max(gris.shape) > a.tam_max:
        f = a.tam_max / max(gris.shape)
        gris = cv2.resize(gris, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

    suave = suavizar(gris, a.suavizado, a.kernel_suavizado)
    binaria = objeto_en_blanco(binarizar(suave, a.umbral, a.bloque_adaptativo,
                                         a.c_adaptativo, a.factor_relativo))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_impar(a.kernel_morfologia),) * 2) # Crea ventana de morfologia 3*3
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, k)    # elimina motas blancas
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, k)   # tapa grietas pequenas

    binaria = quedarse_con_mayor_objeto(binaria, a.area_minima)
    return binaria, gris


def montaje_debug(gris, binaria):
    """Original | binaria, lado a lado, para ver rapido que salio mal."""
    h, w = gris.shape
    der = np.zeros((h, w), np.uint8) if binaria is None else cv2.resize(binaria, (w, h),
                                                                        interpolation=cv2.INTER_NEAREST)
    return np.hstack([gris, np.full((h, 4), 128, np.uint8), der])


def construir_parser():
    p = argparse.ArgumentParser(description="Preprocesamiento a imagenes binarias")
    p.add_argument("-i", "--entrada", default=CARPETA_ENTRADA)
    p.add_argument("-o", "--salida", default=CARPETA_SALIDA)
    p.add_argument("--tam-max", type=int, default=TAM_MAX)
    p.add_argument("--suavizado", choices=["gaussiano", "mediana", "bilateral", "ninguno"],
                   default=SUAVIZADO)
    p.add_argument("--kernel-suavizado", type=int, default=KERNEL_SUAVIZADO)
    p.add_argument("--umbral", choices=["otsu", "adaptativo", "relativo"], default=UMBRAL)
    p.add_argument("--factor-relativo", type=float, default=FACTOR_RELATIVO)
    p.add_argument("--debug", action="store_true", default=DEBUG)
    p.add_argument("--bloque-adaptativo", type=int, default=BLOQUE_ADAPTATIVO)
    p.add_argument("--c-adaptativo", type=int, default=C_ADAPTATIVO)
    p.add_argument("--kernel-morfologia", type=int, default=KERNEL_MORFOLOGIA)
    p.add_argument("--area-minima", type=float, default=AREA_MINIMA)
    p.add_argument("--recortar", action="store_true", default=RECORTAR)
    p.add_argument("--margen-recorte", type=int, default=MARGEN_RECORTE)
    return p


def procesar_carpeta(a):
    """Preprocesa todas las imagenes de a.entrada y las guarda en a.salida.
    Devuelve (cantidad_ok, lista_de_fallidas). Tambien la usa main.py."""
    entrada, salida = Path(a.entrada), Path(a.salida)
    if not entrada.is_dir():
        sys.exit(f"No existe la carpeta de entrada: {entrada}")

    archivos = sorted(f for f in entrada.rglob("*") if f.suffix.lower() in EXTENSIONES)
    if not archivos:
        sys.exit(f"No se encontraron imagenes en {entrada}")

    ok, fallidas = 0, []
    for f in archivos:
        gris = leer_gris(f)
        res, gris_r = preprocesar(gris, a) if gris is not None else (None, None)
        rel = f.relative_to(entrada).with_suffix(".png")
        if a.debug and gris_r is not None:
            guardar_png(salida / "_debug" / rel, montaje_debug(gris_r, res))
        if res is None:
            fallidas.append(f)
            continue
        guardar_png(salida / rel, res)
        ok += 1

    print(f"Procesadas: {ok}/{len(archivos)}  ->  {salida}")
    for f in fallidas:
        print(f"  [!] No se pudo procesar (ilegible o sin objeto claro): {f}")
    return ok, fallidas


def main():
    procesar_carpeta(construir_parser().parse_args())


if __name__ == "__main__":
    main()