"""
K-NN (K vecinos mas cercanos) implementado desde cero, solo con numpy.

    knn = KNN(k=5).fit(X_entrenamiento, etiquetas)
    knn.predict(X_nuevos)           # clase de cada fila
    knn.predict_proba(X_nuevos)     # fraccion de votos por clase
    knn.kneighbors(X_nuevos)        # distancias e indices de los k vecinos

Como funciona: no "aprende" parametros; guarda los ejemplos de entrenamiento. Para clasificar
un audio nuevo mide su distancia a todos los ejemplos, toma los k mas cercanos y la clase
que mas se repite entre ellos gana (voto mayoritario).

Opciones:
    metrica : "euclidea" (por defecto) o "manhattan"
    pesos   : "uniforme" (cada vecino vale 1) o "distancia" (vale mas el que esta mas cerca)
    Desempate: si dos clases tienen los mismos votos, gana la del vecino mas cercano de las empatadas.

Las caracteristicas tienen escalas muy distintas (amplitud 0-1, MFCC de decenas): hay que
estandarizarlas antes (funcion estandarizar) o las de mayor escala dominan la distancia.
"""

import numpy as np


def estandarizar(X, media=None, desvio=None):
    """z-score por caracteristica. Devuelve (X_estandarizada, media, desvio); para datos
    nuevos se reutilizan la media y el desvio del entrenamiento."""
    X = np.asarray(X, dtype=float)
    if media is None:
        media = X.mean(axis=0)
    if desvio is None:
        desvio = X.std(axis=0)
        desvio[desvio == 0] = 1.0            # columnas constantes: evita dividir por cero
    return (X - media) / desvio, media, desvio


class KNN:
    def __init__(self, k=3, metrica="euclidea", pesos="uniforme"):
        if metrica not in ("euclidea", "manhattan"):
            raise ValueError("metrica debe ser 'euclidea' o 'manhattan'")
        if pesos not in ("uniforme", "distancia"):
            raise ValueError("pesos debe ser 'uniforme' o 'distancia'")
        if k < 1:
            raise ValueError("k debe ser al menos 1")
        self.k, self.metrica, self.pesos = int(k), metrica, pesos
        self.X_ = self.y_idx_ = self.clases_ = None

    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(str)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X debe ser (n, d) y y debe tener n etiquetas")
        if self.k > len(X):
            raise ValueError(f"k = {self.k} es mayor que la cantidad de ejemplos ({len(X)})")
        self.X_ = X
        self.clases_ = np.array(sorted(set(y.tolist())))
        self.y_idx_ = np.searchsorted(self.clases_, y)      # clase de cada ejemplo, como indice
        return self

    def _distancias(self, X):
        """Matriz (n_consultas, n_ejemplos)."""
        if self.metrica == "euclidea":
            d2 = ((X ** 2).sum(1)[:, None] - 2.0 * X @ self.X_.T + (self.X_ ** 2).sum(1)[None, :])
            return np.sqrt(np.maximum(d2, 0.0))
        out = np.empty((len(X), len(self.X_)))
        for i in range(0, len(X), 32):                       # por bloques, para no gastar memoria
            out[i:i + 32] = np.abs(X[i:i + 32, None, :] - self.X_[None, :, :]).sum(axis=2)
        return out

    @staticmethod
    def _ordenar(D, k):
        """Los k menores de cada fila, ordenados de menor a mayor: (distancias, indices)."""
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        d = np.take_along_axis(D, idx, axis=1)
        o = np.argsort(d, axis=1)
        return np.take_along_axis(d, o, axis=1), np.take_along_axis(idx, o, axis=1)

    def _decidir(self, dist, idx, k):
        """Voto de los k primeros vecinos (ya ordenados). Devuelve (clase_idx, proporciones)."""
        cls = self.y_idx_[idx[:, :k]]
        w = 1.0 / (dist[:, :k] + 1e-9) if self.pesos == "distancia" else np.ones_like(dist[:, :k])
        votos = np.stack([(w * (cls == c)).sum(axis=1) for c in range(len(self.clases_))], axis=1)
        pred = votos.argmax(axis=1)
        empatadas = votos >= votos.max(axis=1, keepdims=True) - 1e-12
        for i in np.where(empatadas.sum(axis=1) > 1)[0]:    # desempate: el vecino mas cercano
            for j in range(k):
                if empatadas[i, cls[i, j]]:
                    pred[i] = cls[i, j]
                    break
        return pred, votos / votos.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    def kneighbors(self, X, k=None):
        """Devuelve (distancias, indices_de_ejemplos), cada uno de forma (n, k)."""
        if self.X_ is None:
            raise RuntimeError("Primero hay que llamar a fit()")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self._ordenar(self._distancias(X), k or self.k)

    def predict_proba(self, X):
        d, idx = self.kneighbors(X)
        return self._decidir(d, idx, self.k)[1]

    def predict(self, X):
        d, idx = self.kneighbors(X)
        return self.clases_[self._decidir(d, idx, self.k)[0]]

    # ------------------------------------------------------------------
    def guardar(self, ruta, **extra):
        """Guarda el modelo (los ejemplos de entrenamiento + k) y datos extra (arrays o textos)."""
        np.savez(ruta, X_entrenamiento=self.X_, y_entrenamiento=self.clases_[self.y_idx_],
                 k=self.k, metrica=self.metrica, pesos=self.pesos, **extra)

    @classmethod
    def cargar(cls, ruta):
        """Devuelve (knn, extras)."""
        m = np.load(ruta, allow_pickle=False)
        reservadas = {"X_entrenamiento", "y_entrenamiento", "k", "metrica", "pesos"}
        knn = cls(int(m["k"]), str(m["metrica"]), str(m["pesos"]))
        knn.fit(m["X_entrenamiento"], m["y_entrenamiento"])
        return knn, {k: m[k] for k in m.files if k not in reservadas}
