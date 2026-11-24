"""
K-means implementado desde cero.

Pensado para recibir una matriz X de forma (n_muestras, n_caracteristicas),
donde cada fila es el vector de caracteristicas de una imagen.
"""

import numpy as np


def estandarizar(X, media=None, desvio=None):
    """
    Lleva cada caracteristica a media 0 y desvio 1 (z-score).
    Es importante cuando las caracteristicas tienen escalas distintas
    (por ejemplo area en pixeles vs. un valor entre 0 y 1), porque K-means
    usa distancia euclidea y la caracteristica de mayor escala dominaria.

    Devuelve (X_estandarizada, media, desvio) para poder aplicar la misma
    transformacion a imagenes nuevas.
    """
    X = np.asarray(X, dtype=float)
    if media is None:
        media = X.mean(axis=0)
    if desvio is None:
        desvio = X.std(axis=0)
        desvio[desvio == 0] = 1.0  # evita dividir por cero en columnas constantes
    return (X - media) / desvio, media, desvio


class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4,
                 n_init=10, init="k-means++", random_state=None):
        """
        n_clusters   : cantidad de grupos (K).
        max_iter     : maximo de iteraciones por inicializacion.
        tol          : se detiene cuando el desplazamiento total de los
                       centroides es menor a tol.
        n_init       : cuantas veces se corre el algoritmo con centroides
                       iniciales distintos; se queda con la de menor inercia.
        init         : "k-means++", "random", o un arreglo (n_clusters, n_caracteristicas)
                       con los centroides iniciales (por ejemplo, el promedio de piezas
                       ya etiquetadas). Con arreglo, n_init se ignora (una sola corrida).
        random_state : semilla para reproducibilidad.
        """
        if isinstance(init, str):
            if init not in ("k-means++", "random"):
                raise ValueError("init debe ser 'k-means++', 'random' o un arreglo de centroides")
        else:
            init = np.asarray(init, dtype=float)
            if init.ndim != 2 or init.shape[0] != n_clusters:
                raise ValueError("init como arreglo debe tener forma (n_clusters, n_caracteristicas)")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init = init
        self.random_state = random_state

        # Se completan al hacer fit()
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    @staticmethod
    def _distancias_cuadradas(X, centroides):
        """
        Matriz (n_muestras, n_clusters) con la distancia euclidea al cuadrado
        de cada muestra a cada centroide, sin bucles:
        ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2
        """
        x2 = np.sum(X ** 2, axis=1, keepdims=True)
        c2 = np.sum(centroides ** 2, axis=1)
        d2 = x2 - 2.0 * X @ centroides.T + c2
        return np.maximum(d2, 0.0)  # corrige negativos por error numerico

    def _inicializar(self, X, rng):
        n = X.shape[0]
        if not isinstance(self.init, str):
            if self.init.shape[1] != X.shape[1]:
                raise ValueError("Los centroides iniciales tienen otra cantidad de caracteristicas que X")
            return self.init.copy()
        if self.init == "random":
            idx = rng.choice(n, size=self.n_clusters, replace=False)
            return X[idx].copy()

        # k-means++: el primer centroide es al azar; cada uno de los
        # siguientes se elige con probabilidad proporcional a la distancia^2
        # al centroide mas cercano ya elegido.
        centroides = np.empty((self.n_clusters, X.shape[1]))
        centroides[0] = X[rng.integers(n)]
        d2_min = self._distancias_cuadradas(X, centroides[:1]).ravel()
        for k in range(1, self.n_clusters):
            total = d2_min.sum()
            if total == 0:  # todos los puntos restantes coinciden con un centroide
                idx = rng.integers(n)
            else:
                idx = rng.choice(n, p=d2_min / total)
            centroides[k] = X[idx]
            d2_nuevo = self._distancias_cuadradas(X, centroides[k:k + 1]).ravel()
            d2_min = np.minimum(d2_min, d2_nuevo)
        return centroides

    def _una_corrida(self, X, rng):
        centroides = self._inicializar(X, rng)

        for it in range(1, self.max_iter + 1):
            # 1) Asignacion: cada muestra va al centroide mas cercano
            d2 = self._distancias_cuadradas(X, centroides)
            etiquetas = np.argmin(d2, axis=1)

            # 2) Actualizacion: cada centroide pasa a ser la media de su grupo
            nuevos = np.empty_like(centroides)
            for k in range(self.n_clusters):
                miembros = X[etiquetas == k]
                if len(miembros) > 0:
                    nuevos[k] = miembros.mean(axis=0)
                else:
                    # Cluster vacio: lo reubicamos en la muestra que peor
                    # esta representada (la mas lejos de su centroide)
                    dist_propia = d2[np.arange(len(X)), etiquetas]
                    nuevos[k] = X[np.argmax(dist_propia)]

            # 3) Convergencia: cuanto se movieron los centroides
            desplazamiento = np.sum((nuevos - centroides) ** 2)
            centroides = nuevos
            if desplazamiento <= self.tol:
                break

        # Asignacion final con los centroides definitivos
        d2 = self._distancias_cuadradas(X, centroides)
        etiquetas = np.argmin(d2, axis=1)
        inercia = float(np.sum(d2[np.arange(len(X)), etiquetas]))
        return centroides, etiquetas, inercia, it

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X debe tener forma (n_muestras, n_caracteristicas)")
        if X.shape[0] < self.n_clusters:
            raise ValueError("Hay menos muestras que clusters")

        rng = np.random.default_rng(self.random_state)
        mejor = None
        n_corridas = self.n_init if isinstance(self.init, str) else 1
        for _ in range(n_corridas):
            resultado = self._una_corrida(X, rng)
            if mejor is None or resultado[2] < mejor[2]:
                mejor = resultado

        (self.cluster_centers_, self.labels_,
         self.inertia_, self.n_iter_) = mejor
        return self

    def predict(self, X):
        """Asigna cada fila de X al centroide mas cercano (ya entrenado)."""
        if self.cluster_centers_ is None:
            raise RuntimeError("Primero hay que llamar a fit()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.argmin(self._distancias_cuadradas(X, self.cluster_centers_), axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_
