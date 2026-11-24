"""
EXTRACCION DE CARACTERISTICAS DE AUDIO para los comandos de voz.

Recibe los audios que dejo preprocesar_audio.py y divide cada uno en 10 SEGMENTOS iguales.
Para cada segmento calcula:

    ZCR (tasa de cruces por cero)   promedio y maximo
    Amplitud                        promedio y maximo (sobre el valor absoluto de la senal)
    MFCC (coeficientes 1, 2, 4, 5)  promedio, maximo y desviacion estandar de cada uno

Son 4 + 3 x 4 = 16 valores por segmento, o sea 160 caracteristicas por audio.

Detalles de calculo:
  * ZCR y MFCC se calculan por ventanas de 25 ms con salto de 10 ms; el promedio, maximo
    y desvio de cada segmento son sobre las ventanas que le corresponden (las ventanas se
    reparten en 10 grupos consecutivos del mismo tamano). La amplitud se calcula directo
    sobre las muestras de cada segmento.
  * Los MFCC se calculan con numpy: ventana de Hann, espectro de potencia, banco de 26
    filtros triangulares en escala Mel entre 300 y 5000 Hz (la banda que dejo el filtro
    pasa banda), logaritmo en dB y DCT tipo II ortonormal. Se obtienen 13 coeficientes.
  * Numeracion de MFCC: el indice 0 es el coeficiente de energia (c0) y MFCC_INDICES = [1, 2, 4, 5]
    son los indices 1, 2, 4 y 5 en esa numeracion (la que usa librosa). Se cambia en la lista
    de abajo.

Uso suelto (extrae todo y guarda un CSV):
    python caracteristicas_audio.py -i audios_procesados -o resultados_audio
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from preprocesar_audio import leer_wav

# ----------------------------------------------------------------------
# CONFIGURACION
# ----------------------------------------------------------------------
N_SEGMENTOS = 10
MFCC_INDICES = [1, 2, 4, 5]    # 0 = c0 (energia). Ver el encabezado.
N_MFCC = 13
N_MELS = 26
FMIN, FMAX = 300.0, 5000.0     # banda del banco de filtros Mel (la del filtro pasa banda)
VENTANA_MS = 25
SALTO_MS = 10


# ----------------------------------------------------------------------
# MFCC (implementacion propia)
# ----------------------------------------------------------------------
def _hz_a_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, float) / 700.0)


def _mel_a_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, float) / 2595.0) - 1.0)


def banco_mel(fs, nfft, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    """Matriz (n_mels, nfft//2+1) de filtros triangulares espaciados en escala Mel."""
    fmax = min(fmax, fs / 2.0)
    puntos = _mel_a_hz(np.linspace(_hz_a_mel(fmin), _hz_a_mel(fmax), n_mels + 2))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    B = np.zeros((n_mels, len(f)))
    for m in range(n_mels):
        izq, cen, der = puntos[m], puntos[m + 1], puntos[m + 2]
        B[m] = np.maximum(0.0, np.minimum((f - izq) / (cen - izq), (der - f) / (der - cen)))
    return B


def matriz_dct(n_mfcc, n_mels):
    """DCT tipo II ortonormal como matriz (n_mfcc, n_mels)."""
    k = np.arange(n_mfcc)[:, None]
    n = np.arange(n_mels)[None, :]
    D = np.sqrt(2.0 / n_mels) * np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))
    D[0] /= np.sqrt(2.0)
    return D


def _ventanas(x, L, H):
    """Matriz (n_ventanas, L) con las ventanas de la senal."""
    n_v = 1 + (len(x) - L) // H
    return x[np.arange(L)[None, :] + H * np.arange(n_v)[:, None]]


def mfcc_ventanas(x, fs, L, H):
    nfft = max(512, 1 << int(np.ceil(np.log2(L))))
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(L) / L)
    potencia = np.abs(np.fft.rfft(_ventanas(x, L, H) * hann, nfft, axis=1)) ** 2
    mel = potencia @ banco_mel(fs, nfft).T
    log_mel = 10.0 * np.log10(mel + 1e-10)
    return log_mel @ matriz_dct(N_MFCC, N_MELS).T          # (n_ventanas, N_MFCC)


# ----------------------------------------------------------------------
# Caracteristicas
# ----------------------------------------------------------------------
def nombres_caracteristicas():
    nombres = []
    for s in range(1, N_SEGMENTOS + 1):
        nombres += [f"seg{s:02d}_zcr_prom", f"seg{s:02d}_zcr_max",
                    f"seg{s:02d}_amp_prom", f"seg{s:02d}_amp_max"]
        for c in MFCC_INDICES:
            nombres += [f"seg{s:02d}_mfcc{c}_prom", f"seg{s:02d}_mfcc{c}_max", f"seg{s:02d}_mfcc{c}_std"]
    return nombres


def extraer(x, fs):
    """Vector de caracteristicas (N_SEGMENTOS x 16 valores) de un audio ya preprocesado."""
    L = int(fs * VENTANA_MS / 1000)
    H = int(fs * SALTO_MS / 1000)
    minimo = L + (N_SEGMENTOS - 1) * H            # asegura al menos una ventana por segmento
    if len(x) < minimo:
        x = np.pad(x, (0, minimo - len(x)))

    ventanas = _ventanas(x, L, H)
    signo = ventanas >= 0
    zcr = np.mean(signo[:, 1:] != signo[:, :-1], axis=1)       # cruces por cero por ventana
    mfcc = mfcc_ventanas(x, fs, L, H)[:, MFCC_INDICES]

    grupos = np.array_split(np.arange(len(ventanas)), N_SEGMENTOS)   # ventanas de cada segmento
    tramos = np.array_split(x, N_SEGMENTOS)                          # muestras de cada segmento
    v = []
    for g, tramo in zip(grupos, tramos):
        a = np.abs(tramo)
        v += [zcr[g].mean(), zcr[g].max(), a.mean(), a.max()]
        for j in range(len(MFCC_INDICES)):
            v += [mfcc[g, j].mean(), mfcc[g, j].max(), mfcc[g, j].std()]
    return np.array(v, dtype=float)


def extraer_carpeta(carpeta):
    """Recorre los .wav de una carpeta (ignora carpetas que empiezan con '_', como _debug).
    Devuelve X (n, 160), rutas relativas y la clase de cada audio (nombre de la primera
    subcarpeta; None si estan sueltos)."""
    carpeta = Path(carpeta)
    archivos = sorted(f for f in carpeta.rglob("*")
                      if f.suffix.lower() == ".wav"
                      and not any(p.startswith("_") for p in f.relative_to(carpeta).parts))
    X, rutas, clases, frecuencias = [], [], [], set()
    for f in archivos:
        try:
            x, fs = leer_wav(f)
        except (OSError, ValueError, EOFError) as e:
            print(f"  [!] No se pudo leer, se omite ({e}): {f}")
            continue
        if len(x) == 0 or not np.any(x):
            print(f"  [!] Audio vacio, se omite: {f}")
            continue
        frecuencias.add(fs)
        X.append(extraer(x, fs))
        rel = f.relative_to(carpeta)
        rutas.append(str(rel))
        clases.append(rel.parts[0] if len(rel.parts) > 1 else None)
    if len(frecuencias) > 1:
        print(f"  [!] Hay audios con distintas frecuencias de muestreo {sorted(frecuencias)}: "
              "las caracteristicas no son comparables. Preprocesa todo con la misma --fs.")
    return np.array(X), rutas, clases


def guardar_csv(ruta, X, rutas, clases):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["archivo", "clase"] + nombres_caracteristicas())
        for r, c, fila in zip(rutas, clases, X):
            w.writerow([r, c or ""] + [f"{v:.6g}" for v in fila])


# ----------------------------------------------------------------------
# Uso suelto
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Extraccion de caracteristicas de audio")
    p.add_argument("-i", "--entrada", default="audios_procesados")
    p.add_argument("-o", "--salida", default="resultados_audio")
    a = p.parse_args()

    if not Path(a.entrada).is_dir():
        sys.exit(f"No existe la carpeta: {a.entrada}")
    X, rutas, clases = extraer_carpeta(a.entrada)
    if len(X) == 0:
        sys.exit("No se encontraron audios .wav validos.")
    destino = Path(a.salida) / "caracteristicas_audio.csv"
    guardar_csv(destino, X, rutas, clases)
    print(f"{len(X)} audios, {X.shape[1]} caracteristicas por audio  ->  {destino}")
    if all(c is not None for c in clases):
        print("Audios por clase:", dict(Counter(clases)))


if __name__ == "__main__":
    main()
