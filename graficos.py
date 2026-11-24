"""
GRAFICOS de la interfaz, dibujados como imagenes (PIL.Image) con matplotlib.

No usan pyplot ni el backend de tkinter: cada funcion arma una Figure, la dibuja con el
backend Agg y devuelve una imagen del tamano pedido en pixeles. La ventana solo tiene que
mostrarla. Asi se pueden generar y revisar sin abrir ninguna ventana.
"""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

import bayes as B

AZUL, GRIS, NARANJA, VERDE, ROJO = "#2a6fbb", "#b8c2d0", "#e08a1e", "#3a9a4b", "#c0392b"
DPI = 100


def _figura(ancho, alto):
    return Figure(figsize=(ancho / DPI, alto / DPI), dpi=DPI, facecolor="white")


def _a_imagen(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3].copy())


def _estilo(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def img_conteo(conteos, ancho=480, alto=250):
    """Barras con las piezas de cada tipo reconocidas en la muestra (comando 'Contar')."""
    fig = _figura(ancho, alto)
    ax = fig.add_subplot(111)
    total = max(sum(conteos), 1)
    barras = ax.barh(range(len(B.CLASES)), conteos, color=AZUL, edgecolor="k", linewidth=0.5)
    for b, n in zip(barras, conteos):
        ax.text(b.get_width() + 0.08, b.get_y() + b.get_height() / 2, f"{n}  ({n / total * 100:.0f} %)",
                va="center", fontsize=9)
    ax.set_yticks(range(len(B.CLASES)))
    ax.set_yticklabels(B.CLASES, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(conteos), 1) * 1.35)
    ax.set_xlabel("piezas reconocidas", fontsize=8)
    ax.set_title(f"Piezas reconocidas en la muestra ({sum(conteos)})", fontsize=10)
    _estilo(ax)
    ax.grid(axis="y", alpha=0)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _a_imagen(fig)


def img_estimacion(est, ancho=480, alto=430, caja_real=None):
    """Comando 'Proporcion': probabilidad de cada caja y composicion estimada de las 1000 piezas."""
    fig = _figura(ancho, alto)
    ax1, ax2 = fig.subplots(2, 1, gridspec_kw={"height_ratios": [1, 1.25]})

    colores = [AZUL if c == est.caja_map else GRIS for c in est.cajas]
    barras = ax1.bar(range(len(est.cajas)), est.posterior * 100, color=colores, edgecolor="k", linewidth=0.5)
    for b, p in zip(barras, est.posterior):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{p * 100:.1f} %", ha="center", fontsize=9)
    ax1.axhline(100 / len(est.cajas), color="k", linestyle="--", linewidth=0.9)
    ax1.text(len(est.cajas) - 0.5, 100 / len(est.cajas) + 2, "a priori 25 %", ha="right", fontsize=7)
    ax1.set_xticks(range(len(est.cajas)))
    ax1.set_xticklabels([f"caja {c}" + ("\n(real)" if c == caja_real else "") for c in est.cajas], fontsize=8)
    ax1.set_ylim(0, 118)
    ax1.set_ylabel("probabilidad (%)", fontsize=8)
    ax1.set_title("Probabilidad de cada caja", fontsize=10)
    _estilo(ax1)

    n = max(sum(est.conteos), 1)
    series = [("muestra x 100", np.array(est.conteos) / n * 1000, GRIS),
              ("estimación", est.esperado, AZUL),
              (f"caja {est.caja_map} (la más probable)", np.array(est.map_piezas), NARANJA)]
    if caja_real:
        series.append((f"caja real ({caja_real})", np.array(B.CAJAS[caja_real]), VERDE))
    ancho_barra = 0.8 / len(series)
    x = np.arange(len(B.CLASES))
    for j, (nombre, vals, col) in enumerate(series):
        ax2.bar(x + (j - (len(series) - 1) / 2) * ancho_barra, vals, ancho_barra, label=nombre,
                color=col, edgecolor="k", linewidth=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(B.CLASES, fontsize=8)
    ax2.set_ylabel("piezas (de 1000)", fontsize=8)
    ax2.set_title("Composición estimada de la caja completa", fontsize=10)
    ax2.set_ylim(0, max(max(s[1].max() for s in series), 1) * 1.55)     # lugar para la leyenda
    ax2.legend(fontsize=7, loc="upper center", ncol=2, framealpha=0.9)
    _estilo(ax2)
    fig.tight_layout()
    return _a_imagen(fig)


def img_audio(res, ancho=480, alto=330):
    """Las tres etapas del preprocesamiento de un audio (res es lo que devuelve
    preprocesar_audio.preprocesar): original, filtrado con lo que se conserva, y final."""
    fs, m = res["fs"], res["mascara"]
    fig = _figura(ancho, alto)
    axs = fig.subplots(3, 1, sharex=False)
    t = np.arange(len(res["original"])) / fs
    axs[0].plot(t, res["original"], lw=0.5, color=AZUL)
    axs[0].set_title("Grabación original", fontsize=9)
    axs[1].plot(t, res["filtrada"], lw=0.5, color=AZUL)
    d = np.diff(np.r_[0, m.astype(int), 0])
    for ini, fin in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        axs[1].axvspan(ini / fs, fin / fs, color=VERDE, alpha=0.25)
    axs[1].set_title("Filtrada 300-5000 Hz (verde = tramo que se conserva)", fontsize=9)
    axs[2].plot(np.arange(len(res["senal"])) / fs, res["senal"], lw=0.5, color=NARANJA)
    axs[2].set_ylim(-1.05, 1.05)
    axs[2].set_title("Recortada y normalizada", fontsize=9)
    axs[2].set_xlabel("tiempo (s)", fontsize=8)
    for ax in axs:
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    fig.tight_layout()
    return _a_imagen(fig)


def img_distancias(distancias, ancho=360, alto=190):
    """Distancia de una pieza a los 4 centroides de K-means (la menor es la clase elegida)."""
    clases = list(distancias)
    d = np.array([distancias[c] for c in clases])
    fig = _figura(ancho, alto)
    ax = fig.add_subplot(111)
    colores = [AZUL if v == d.min() else GRIS for v in d]
    barras = ax.barh(range(len(clases)), d, color=colores, edgecolor="k", linewidth=0.5)
    for b, v in zip(barras, d):
        ax.text(b.get_width() + d.max() * 0.02, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=8)
    ax.set_yticks(range(len(clases)))
    ax.set_yticklabels(clases, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, d.max() * 1.2)
    ax.set_title("Distancia a cada centroide", fontsize=9)
    ax.tick_params(labelsize=7)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    return _a_imagen(fig)