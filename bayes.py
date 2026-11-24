"""
ESTIMACION BAYESIANA de la composicion de una caja a partir de una muestra de piezas.

Escenario: hay 4 tipos de caja (a, b, c, d), cada una con 1000 piezas en distintas
proporciones. El sistema elige UNA caja al azar y extrae 10 piezas; con
la regla de Bayes se calcula que tan probable es cada caja:

    P(caja | muestra)  =  P(muestra | caja) * P(caja)  /  P(muestra)

    P(caja)            a priori: 1/4 cada una (se elige al azar)
    P(muestra | caja)  verosimilitud: probabilidad de sacar justo esa muestra de esa caja.
                       Las piezas se extraen SIN reponer, pero la muestra es del 1%, se usa distribucion
                       multinomial

Estimacion de la caja completa:
    * caja mas probable (MAP) y su probabilidad
    * composicion esperada: promedio de las 4 cajas ponderado por su probabilidad posterior
      (es la estimacion de menor error cuadratico medio)
"""


import math
import unicodedata
from dataclasses import dataclass

import numpy as np

CLASES = ("tornillos", "clavos", "arandelas", "tuercas")

CAJAS = {
    "a": (250, 250, 250, 250),
    "b": (150, 300, 300, 250),
    "c": (250, 350, 250, 150),
    "d": (500, 500, 0, 0),
}

def normalizar_clase(nombre):
    """'Tuerca', 'tuercas ', 'TORNILLOS'... -> nombre canonico de CLASES."""
    s = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode().lower().strip()
    for c in CLASES:
        if s in (c, c[:-1]):
            return c
    raise ValueError(f"Clase desconocida: '{nombre}' (se esperaba una de {', '.join(CLASES)})")

def conteos_desde_lista(clases_predichas):
    """Lista de clases (una por pieza) -> tupla de conteos en el orden de CLASES."""
    c = [normalizar_clase(x) for x in clases_predichas]
    return tuple(c.count(k) for k in CLASES)

# Matematica
def composiciones(n, k=4):
    """Todas las formas de repartir n piezas en k tipos (tuplas que suman n)."""
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for resto in composiciones(n - i, k - 1):
            yield (i,) + resto


def prob_multinomial(m, caja):
    n, total = sum(m), sum(caja)
    p = math.factorial(n)
    for mi, ki in zip(m, caja):
        p = p / math.factorial(mi) * (ki / total) ** mi
    return p


def matriz_verosimilitud(n, cajas=CAJAS):
    """L[caja, muestra] = P(muestra | caja)."""

    comps = list(composiciones(n))

    H = np.array([
        [prob_multinomial(m, caja) for m in comps]
        for caja in cajas.values()
    ])

    return comps, H

# Estimacion
@dataclass
class Estimacion:
    conteos: tuple            # piezas de cada clase en la muestra (orden de CLASES)
    cajas: list               # nombres de las cajas
    prior: np.ndarray
    verosimilitud: np.ndarray
    posterior: np.ndarray
    caja_map: str             # caja mas probable
    p_map: float
    esperado: np.ndarray      # piezas esperadas de cada clase en la caja (promedio ponderado)
    map_piezas: tuple         # piezas de cada clase de la caja mas probable


def estimar(conteos, prior=None, cajas=CAJAS):
    """Regla de Bayes sobre las cajas. 'conteos' en el orden de CLASES."""
    conteos = tuple(int(c) for c in conteos)
    n = sum(conteos)
    if n == 0:
        raise ValueError("La muestra esta vacia")
    nombres = list(cajas)
    prior = np.full(len(nombres), 1.0 / len(nombres)) if prior is None else np.asarray(prior, float)
    comps, L = matriz_verosimilitud(n, cajas)
    lik = L[:, comps.index(conteos)]
    conjunta = prior * lik
    if conjunta.sum() <= 0:
        raise ValueError("Esa muestra es imposible bajo todas las cajas")
    post = conjunta / conjunta.sum()
    i = int(post.argmax())
    esperado = post @ np.array([cajas[c] for c in nombres], float)
    return Estimacion(conteos, nombres, prior, lik, post, nombres[i], float(post[i]), esperado, cajas[nombres[i]])


# ----------------------------------------------------------------------
# Textos (los usara el comando de voz)
# ----------------------------------------------------------------------
def texto_conteo(conteos):
    """Lo que responde el comando 'Contar': piezas de cada tipo reconocidas en la muestra."""
    return "Piezas reconocidas en la muestra (%d): %s" % (
        sum(conteos), ", ".join(f"{c} {n}" for c, n in zip(CLASES, conteos)))


def texto_proporcion(est, cajas=CAJAS):
    """Lo que responde el comando 'Proporcion': probabilidad de cada caja y composicion estimada."""
    total = sum(cajas[est.caja_map])
    lineas = ["Probabilidad de cada caja:"]
    for nombre, p, lik in zip(est.cajas, est.posterior, est.verosimilitud):
        lineas.append("  caja %s  (%-17s)  %5.1f %%" % (
            nombre, " / ".join(str(v) for v in cajas[nombre]), p * 100))
    lineas.append("Caja mas probable: %s (%.1f %%)" % (est.caja_map, est.p_map * 100))
    lineas.append("Composicion estimada de la caja (promedio ponderado por probabilidad):")
    for c, e, m in zip(CLASES, est.esperado, est.map_piezas):
        lineas.append("  %-10s %6.1f piezas  (%4.1f %%)   | caja mas probable: %d" % (c, e, e / total * 100, m))
    return "\n".join(lineas)
