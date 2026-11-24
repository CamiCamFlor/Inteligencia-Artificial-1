"""
PREPROCESAMIENTO DE AUDIO para los comandos de voz ("proporcion", "contar", "salir").

Lee los .wav de una carpeta (incluye subcarpetas) y guarda en otra la version procesada.
Respeta la estructura de subcarpetas: si ordenas las grabaciones por comando
(audios_originales/contar/, /proporcion/, /salir/) la salida queda igual.

Pipeline, en este orden:
    0. Lectura, paso a mono y (opcional) remuestreo a una frecuencia comun
    1. FILTRO PASA BANDA: atenua por debajo de 300 Hz y por encima de 5000 Hz
    2. ELIMINACION DE SILENCIOS: se descartan los tramos cuyo nivel esta mas de 25 dB
       por debajo del tramo mas fuerte del audio
    3. NORMALIZACION: se divide por el valor absoluto maximo (queda entre -1 y 1)

Solo usa numpy y la libreria estandar.

Uso:
    python preprocesar_audio.py                           # usa las carpetas de abajo
    python preprocesar_audio.py -i grabaciones -o procesados --debug
    python preprocesar_audio.py --silencios todos --umbral-db 30
"""

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

CARPETA_ENTRADA = "Audios_Entrenamiento"
CARPETA_SALIDA = "audios_procesados"

F_BAJO = 300.0        # Hz: por debajo se atenua
F_ALTO = 5000.0       # Hz: por encima se atenua
ORDEN = 4             # orden del filtro Butterworth (mayor = corte mas abrupto)
UMBRAL_DB = 25.0      # se descarta lo que esta mas de 25 dB por debajo del pico
SILENCIOS = "extremos"  # "extremos": recorta inicio y final | "todos": saca tambien las pausas internas
FS_OBJETIVO = 16000   # Hz; 0 = no remuestrear. Conviene que todos los audios tengan la misma
VENTANA_MS = 25       # ventana para medir el nivel (dB)
SALTO_MS = 10
DURACION_MIN_S = 0.10  # si queda menos que esto despues de recortar, se descarta el audio

PARAMETROS = ["f_bajo", "f_alto", "orden", "umbral_db", "silencios", "fs", "ventana_ms", "salto_ms"]
OTROS_FORMATOS = {".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".wma", ".mp4", ".amr"}


# ----------------------------------------------------------------------
# Lectura y escritura de WAV 
# ----------------------------------------------------------------------
def leer_wav(ruta):
    """Devuelve (senal_mono_float64 entre -1 y 1, frecuencia_de_muestreo)."""
    try:
        with wave.open(str(ruta), "rb") as w:
            nch, sw, fs, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            datos = w.readframes(n)
    except wave.Error as e:
        # WAV en coma flotante u otros formatos: se intenta con scipy si esta instalado
        try:
            from scipy.io import wavfile
        except ImportError:
            raise IOError(f"formato WAV no soportado ({e}); guardalo como PCM de 16 bits")
        fs, d = wavfile.read(str(ruta))
        escala = float(np.iinfo(d.dtype).max + 1) if np.issubdtype(d.dtype, np.integer) else 1.0
        d = d.astype(np.float64) / escala
        return (d if d.ndim == 1 else d.mean(axis=1)), fs

    if sw == 1:
        x = (np.frombuffer(datos, np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sw == 2:
        x = np.frombuffer(datos, "<i2").astype(np.float64) / 32768.0
    elif sw == 3:
        b = np.frombuffer(datos, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v & 0x800000, v - 0x1000000, v).astype(np.float64) / 8388608.0
    elif sw == 4:
        x = np.frombuffer(datos, "<i4").astype(np.float64) / 2147483648.0
    else:
        raise IOError(f"profundidad de {sw * 8} bits no soportada")
    return x.reshape(-1, nch).mean(axis=1), fs      # varios canales -> promedio (mono)


def guardar_wav(ruta, x, fs):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(pcm.tobytes())


# ----------------------------------------------------------------------
# Etapas del procesamiento
# ----------------------------------------------------------------------
def remuestrear(x, fs, fs_obj):
    """Remuestreo exacto por FFT (recorta o rellena el espectro)."""
    if not fs_obj or int(fs) == int(fs_obj):
        return x, fs
    n = len(x)
    m = int(round(n * fs_obj / fs))
    X = np.fft.rfft(x)
    Y = np.zeros(m // 2 + 1, dtype=complex)
    k = min(len(X), len(Y))
    Y[:k] = X[:k]
    return np.fft.irfft(Y, m) * (m / n), fs_obj


def respuesta_pasa_banda(f, f_bajo, f_alto, orden):
    """Modulo de la respuesta de un Butterworth pasa banda: pasa-altos en f_bajo por
    pasa-bajos en f_alto. En ambas frecuencias de corte la ganancia es -3 dB."""
    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore"):
        h_alto = 1.0 / np.sqrt(1.0 + (f_bajo / f) ** (2 * orden))    # en f = 0 vale 0
    h_bajo = 1.0 / np.sqrt(1.0 + (f / f_alto) ** (2 * orden))
    return h_alto * h_bajo


def pasa_banda(x, fs, f_bajo, f_alto, orden):
    """Filtro aplicado en frecuencia (fase cero: no desplaza ni deforma la forma de onda)."""
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(max(2 * n, 2))))     # relleno con ceros para evitar el efecto de borde
    X = np.fft.rfft(x, nfft)
    H = respuesta_pasa_banda(np.fft.rfftfreq(nfft, 1.0 / fs), f_bajo, f_alto, orden)
    return np.fft.irfft(X * H, nfft)[:n]


def detectar_voz(x, fs, umbral_db, ventana_ms, salto_ms, modo):
    """Mascara booleana (una por muestra) con True donde hay sonido.
    El nivel de cada ventana de 25 ms se mide en dB respecto de la ventana mas fuerte
    (0 dB = lo mas fuerte); se descarta lo que cae por debajo de -umbral_db."""
    L = max(1, int(fs * ventana_ms / 1000))
    H = max(1, int(fs * salto_ms / 1000))
    n = len(x)
    xp = np.pad(x, (0, max(0, L - n)))
    n_v = 1 + (len(xp) - L) // H
    idx = np.arange(L)[None, :] + H * np.arange(n_v)[:, None]
    rms = np.sqrt(np.mean(xp[idx] ** 2, axis=1))
    if rms.max() < 1e-9:
        return None                                       # silencio total
    db = 20.0 * np.log10(np.maximum(rms, 1e-12) / rms.max())
    activa = np.where(db > -umbral_db)[0]

    mascara = np.zeros(len(xp), bool)
    if modo == "extremos":
        mascara[activa[0] * H: activa[-1] * H + L] = True
    else:                                                 # "todos": se saca cada tramo silencioso
        for i in activa:
            mascara[i * H: i * H + L] = True
    return mascara[:n]


def preprocesar(x, fs, a):
    """Devuelve (resultado, motivo_de_falla). resultado es un diccionario con la senal
    final normalizada y las etapas intermedias (para el grafico de depuracion)."""
    if len(x) == 0:
        return None, "archivo vacio"
    x, fs = remuestrear(x, fs, a.fs)
    if a.f_alto >= fs / 2:
        print(f"      [!] La frecuencia de muestreo ({fs} Hz) no llega a {a.f_alto:.0f} Hz: "
              "el filtro pasa-bajos no tiene efecto")
    filtrada = pasa_banda(x, fs, a.f_bajo, a.f_alto, a.orden)

    mascara = detectar_voz(filtrada, fs, a.umbral_db, a.ventana_ms, a.salto_ms, a.silencios)
    if mascara is None:
        return None, "solo silencio"
    recortada = filtrada[mascara]
    if len(recortada) < DURACION_MIN_S * fs:
        return None, f"queda menos de {DURACION_MIN_S} s despues de recortar"

    pico = np.max(np.abs(recortada))
    return dict(senal=recortada / pico, fs=fs, original=x, filtrada=filtrada, mascara=mascara), None


def guardar_debug(ruta_png, res):
    """Grafico: original, filtrada (en verde lo que se conserva) y resultado final."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs, m = res["fs"], res["mascara"]
    t = np.arange(len(res["original"])) / fs
    fig, axs = plt.subplots(3, 1, figsize=(10, 7))
    axs[0].plot(t, res["original"], lw=0.5)
    axs[0].set_title("Original")
    axs[1].plot(t, res["filtrada"], lw=0.5)
    d = np.diff(np.r_[0, m.astype(int), 0])
    for ini, fin in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        axs[1].axvspan(ini / fs, fin / fs, color="green", alpha=0.25)
    axs[1].set_title("Filtrada (verde = tramo que se conserva)")
    axs[2].plot(np.arange(len(res["senal"])) / fs, res["senal"], lw=0.5)
    axs[2].set_ylim(-1.05, 1.05)
    axs[2].set_title("Final: recortada y normalizada")
    for ax in axs:
        ax.set_xlabel("tiempo (s)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_png, dpi=90)
    plt.close(fig)


# ----------------------------------------------------------------------
# Carpetas y linea de comandos
# ----------------------------------------------------------------------
def construir_parser():
    p = argparse.ArgumentParser(description="Preprocesamiento de audio (filtro, silencios, normalizacion)")
    p.add_argument("-i", "--entrada", default=CARPETA_ENTRADA)
    p.add_argument("-o", "--salida", default=CARPETA_SALIDA)
    p.add_argument("--f-bajo", type=float, default=F_BAJO, help="Hz, corte inferior")
    p.add_argument("--f-alto", type=float, default=F_ALTO, help="Hz, corte superior")
    p.add_argument("--orden", type=int, default=ORDEN, help="orden del Butterworth")
    p.add_argument("--umbral-db", type=float, default=UMBRAL_DB,
                   help="se descarta lo que esta mas de X dB por debajo del pico")
    p.add_argument("--silencios", choices=["extremos", "todos"], default=SILENCIOS)
    p.add_argument("--fs", type=int, default=FS_OBJETIVO, help="remuestrear a esta frecuencia (0 = no)")
    p.add_argument("--ventana-ms", type=float, default=VENTANA_MS)
    p.add_argument("--salto-ms", type=float, default=SALTO_MS)
    p.add_argument("--debug", action="store_true",
                   help="guarda un grafico por audio en salida/_debug")
    return p


def procesar_carpeta(a):
    """Procesa todos los .wav de a.entrada hacia a.salida. Devuelve (cantidad_ok, fallidas).
    Tambien la va a usar el programa principal."""
    entrada, salida = Path(a.entrada), Path(a.salida)
    if not entrada.is_dir():
        sys.exit(f"No existe la carpeta de entrada: {entrada}")

    todos = sorted(f for f in entrada.rglob("*") if f.is_file())
    archivos = [f for f in todos if f.suffix.lower() == ".wav"]
    otros = [f for f in todos if f.suffix.lower() in OTROS_FORMATOS]
    if otros:
        print(f"[!] {len(otros)} archivo(s) que no son .wav se ignoran (convertilos a WAV, por ejemplo "
              f"con Audacity): {', '.join(f.name for f in otros[:5])}{'...' if len(otros) > 5 else ''}")
    if not archivos:
        sys.exit(f"No se encontraron archivos .wav en {entrada}")

    ok, fallidas, dur_in, dur_out = 0, [], [], []
    for f in archivos:
        rel = f.relative_to(entrada)
        try:
            x, fs = leer_wav(f)
            res, motivo = preprocesar(x, fs, a)
        except (OSError, ValueError, EOFError, wave.Error) as e:
            res, motivo = None, str(e)
        if res is None:
            fallidas.append((f, motivo))
            continue
        guardar_wav(salida / rel.with_suffix(".wav"), res["senal"], res["fs"])
        if a.debug:
            guardar_debug(salida / "_debug" / rel.with_suffix(".png"), res)
        dur_in.append(len(x) / fs)
        dur_out.append(len(res["senal"]) / res["fs"])
        ok += 1

    # Se guardan los parametros: las grabaciones nuevas deben procesarse IGUAL
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "_preproceso_audio.json").write_text(
        json.dumps({k: getattr(a, k) for k in PARAMETROS}, indent=2), encoding="utf-8")

    print(f"Procesados: {ok}/{len(archivos)}  ->  {salida}")
    if ok:
        print(f"Duracion media: {np.mean(dur_in):.2f} s originales -> {np.mean(dur_out):.2f} s procesados")
    for f, motivo in fallidas:
        print(f"  [!] No se pudo procesar ({motivo}): {f}")
    return ok, fallidas


def main():
    procesar_carpeta(construir_parser().parse_args())


if __name__ == "__main__":
    main()
