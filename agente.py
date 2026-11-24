"""
NUCLEO DEL SISTEMA (sin ventanas): une todas las etapas del trabajo.

    caja al azar  ->  muestra de 10 piezas  ->  fotos identificadas con K-means
                                              \\->  estimacion bayesiana de la caja
    comando de voz (K-NN): "proporcion", "contar", "salir"

La interfaz (gui.py) solo muestra lo que decide esta clase. Asi la logica se puede probar sin
abrir ventanas (probar_agente.py) y las reglas de "que se puede hacer y cuando" estan en un
solo lugar, el metodo disponible():

    sortear_caja()             siempre
    extraer_muestra()          necesita caja
    identificar_pieza(i, ..)   necesita muestra
    "contar"                   necesita las 10 piezas identificadas
    "proporcion"               necesita las 10 piezas identificadas
    "salir"                    siempre

Los modelos se leen de los .npz que dejan main.py (K-means, imagenes) y entrenar_audio.py
(K-NN, voz). Cada modelo trae guardados los parametros de preprocesamiento con que se
entreno, y se aplican identicos a las fotos y audios nuevos.
"""

import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np

import bayes as B
import caracteristicas as car
import caracteristicas_audio as cau
import preprocesar as pre
import preprocesar_audio as pau
from knn import KNN

N_PIEZAS = 10
EXT_IMAGEN = pre.EXTENSIONES
COMANDOS = ("proporcion", "contar", "salir")
UMBRAL_CONFIANZA = 0.5      # fraccion minima de votos de K-NN para aceptar un comando
CONTRASTE_MIN_DB = 10.0     # grabacion en vivo: la palabra debe sobresalir al menos esto del ruido de fondo


def _sin_tildes(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def _clase_desde_nombre(nombre):
    """'tuercas' -> 'tuercas'. Tambien entiende 'Cluster 2 (~tuercas)', que deja main.py cuando
    los clusters se nombran por mayoria."""
    try:
        return B.normalizar_clase(nombre)
    except ValueError:
        m = re.search(r"~\s*([^)\s]+)", str(nombre))
        if m:
            return B.normalizar_clase(m.group(1))
        raise


def contraste_db(senal, fs, ventana_ms=25, salto_ms=10):
    """Cuanto sobresale la ventana mas fuerte del nivel tipico (mediana) de la grabacion, en dB.
    Sirve para saber si en una grabacion en vivo se dijo algo o solo hay ruido de fondo."""
    L = max(1, int(fs * ventana_ms / 1000))
    H = max(1, int(fs * salto_ms / 1000))
    if len(senal) < L:
        return 0.0
    idx = np.arange(L)[None, :] + H * np.arange(1 + (len(senal) - L) // H)[:, None]
    db = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(senal[idx] ** 2, axis=1)), 1e-9))
    return float(db.max() - np.median(db))


class ErrorSistema(Exception):
    """Algo que el usuario puede corregir (falta un modelo, una foto ilegible, etc.)."""


class Agente:
    def __init__(self, base=None, semilla=None):
        self.base = Path(base) if base else Path(__file__).resolve().parent
        self.rng = np.random.default_rng(semilla)
        self.registro = []                      # (hora, texto)
        self.ruta_sesion = None
        self.kmeans_listo = False
        self.knn_listo = False
        self.ruta_kmeans = self.base / "resultados" / "modelo_kmeans.npz"
        self.ruta_knn = self.base / "resultados_audio" / "modelo_knn_audio.npz"
        self.reiniciar()

    # ------------------------------------------------------------------
    # Modelos
    # ------------------------------------------------------------------
    def cargar_modelos(self):
        """Intenta cargar los dos modelos en sus carpetas habituales. Devuelve la lista de avisos
        (vacia si todo salio bien); no lanza error para que la interfaz pueda abrirse igual."""
        avisos = []
        for cargar, ruta in ((self.cargar_kmeans, self.ruta_kmeans), (self.cargar_knn, self.ruta_knn)):
            try:
                cargar(ruta)
            except ErrorSistema as e:
                avisos.append(str(e))
        return avisos

    def cargar_kmeans(self, ruta):
        ruta = Path(ruta)
        self.kmeans_listo = False
        if not ruta.is_file():
            raise ErrorSistema(f"Falta el modelo de imagenes: {ruta}\nEntrenalo en la pestana Entrenamiento (o con main.py).")
        try:
            m = np.load(ruta, allow_pickle=False)
            centroides, media, desvio = m["centroides"], m["media"], m["desvio"]
            nombres_car = [str(n) for n in m["caracteristicas"]]
            clases = [_clase_desde_nombre(n) for n in m["nombres_clases"]]
            epsilon = float(m["epsilon_vertices"])
            texto = str(m["parametros_preproceso"])
        except KeyError as e:
            raise ErrorSistema(f"{ruta.name} es de una version anterior (le falta {e}). Reentrena el modelo de imagenes.")
        except ValueError as e:
            raise ErrorSistema(f"El modelo de imagenes no tiene las clases esperadas: {e}\n"
                               "Entrena con las imagenes en subcarpetas por clase (tornillos, clavos, arandelas, tuercas).")
        if sorted(clases) != sorted(B.CLASES):
            raise ErrorSistema("El modelo de imagenes debe tener las 4 clases: " + ", ".join(B.CLASES)
                               + f". Tiene: {clases}")
        self.centroides, self.media, self.desvio = centroides, media, desvio
        self.nombres_car, self.clases_cluster = nombres_car, clases
        car.EPSILON_VERTICES = epsilon
        self.params_img = pre.construir_parser().parse_args([])
        if texto:
            for k, v in json.loads(texto).items():
                setattr(self.params_img, k, v)
        self.ruta_kmeans = ruta
        self.kmeans_listo = True
        self.est = None
        self.log(f"Modelo de imagenes (K-means) cargado: {ruta.name}")

    def cargar_knn(self, ruta):
        ruta = Path(ruta)
        self.knn_listo = False
        if not ruta.is_file():
            raise ErrorSistema(f"Falta el modelo de voz: {ruta}\nEntrenalo en la pestana Entrenamiento (o con entrenar_audio.py).")
        try:
            knn, extra = KNN.cargar(ruta)
            media_a, desvio_a = extra["media"], extra["desvio"]
        except KeyError as e:
            raise ErrorSistema(f"{ruta.name} es de una version anterior (le falta {e}). Reentrena el modelo de voz.")
        params = pau.construir_parser().parse_args([])
        texto = str(extra.get("parametros_preproceso", ""))
        if texto:
            for k, v in json.loads(texto).items():
                setattr(params, k, v)
        # Las caracteristicas de audio dependen de constantes de caracteristicas_audio.py:
        # tienen que ser las mismas que al entrenar.
        if "parametros_caracteristicas" in extra:
            for k, v in json.loads(str(extra["parametros_caracteristicas"])).items():
                actual = getattr(cau, k)
                if list(np.ravel(actual)) != list(np.ravel(v)):
                    raise ErrorSistema(f"caracteristicas_audio.py tiene {k} = {actual} pero el modelo de voz "
                                       f"se entreno con {v}. Usa la misma version del archivo o reentrena.")
        comandos = [_sin_tildes(c) for c in knn.clases_]
        faltan = [c for c in COMANDOS if c not in comandos]
        if faltan:
            raise ErrorSistema(f"El modelo de voz no tiene los comandos: {', '.join(faltan)} "
                               f"(tiene: {', '.join(comandos)})")
        self.knn, self.media_a, self.desvio_a, self.params_aud = knn, media_a, desvio_a, params
        self.comandos_modelo = comandos
        self.ruta_knn = ruta
        self.knn_listo = True
        self.log(f"Modelo de voz (K-NN, k = {knn.k}) cargado: {ruta.name}")

    # ------------------------------------------------------------------
    # Estado y registro
    # ------------------------------------------------------------------
    def reiniciar(self):
        self.piezas = []       # dict(ruta, clase, distancia, binaria, detalle)
        self.ultimo_comando = None
        self.est = None

    def log(self, texto):
        self.registro.append((time.strftime("%H:%M:%S"), texto))
        if self.ruta_sesion:
            with open(self.ruta_sesion, "a", encoding="utf-8") as fh:
                fh.write(f"[{self.registro[-1][0]}] {texto}\n")

    def iniciar_registro(self, carpeta=None):
        carpeta = Path(carpeta) if carpeta else self.base / "sesiones"
        carpeta.mkdir(parents=True, exist_ok=True)
        self.ruta_sesion = carpeta / time.strftime("sesion_%Y%m%d_%H%M%S.txt")
        with open(self.ruta_sesion, "w", encoding="utf-8") as fh:
            for h, t in self.registro:
                fh.write(f"[{h}] {t}\n")

    @property
    def modelos_listos(self):
        return self.kmeans_listo and self.knn_listo

    @property
    def hay_muestra(self):
        return len(self.piezas) > 0

    @property
    def n_identificadas(self):
        return sum(1 for p in self.piezas if p["clase"] is not None)

    @property
    def muestra_completa(self):
        return len(self.piezas) == N_PIEZAS and self.n_identificadas == N_PIEZAS

    def cargar_imagenes(self, rutas):
        """Prepara las 10 imágenes seleccionadas para identificarlas con K-means."""
        rutas = [Path(r) for r in rutas]
        if len(rutas) != N_PIEZAS:
            raise ErrorSistema(f"Debes seleccionar exactamente {N_PIEZAS} imágenes.")
        if not self.kmeans_listo:
            raise ErrorSistema("Falta cargar el modelo de imágenes (K-means).")
        self.piezas = [
            dict(ruta=str(r), clase=None, distancia=None, binaria=None, detalle=None)
            for r in rutas
        ]
        self.est = None
        self.ultimo_comando = None
        self.log(f"Se cargaron {N_PIEZAS} imágenes para identificar.")

    def conteos(self):
        return B.conteos_desde_lista([p["clase"] for p in self.piezas if p["clase"] is not None])

    def disponible(self, comando):
        """Indica si un comando puede ejecutarse."""
        # Salir siempre está disponible.
        if comando == "salir":
            return True, ""
        if comando not in ("contar", "proporcion"):
            return False, f"Comando desconocido: {comando}"
        if not self.muestra_completa:
            return False, (
                f"Primero hay que identificar las 10 imágenes. "
                f"Identificadas: {self.n_identificadas} de {N_PIEZAS}."
            )
        # Además de estar identificadas, Bayes debe haberse calculado
        # para los conteos actuales.
        if self.est is None or self.est.conteos != self.conteos():
            return False, "Primero hay que calcular la estimación bayesiana."
        return True, ""
    
    def pasos(self):
        """[(clave, etiqueta, estado)] con estado 'hecho', 'actual' o 'bloqueado'."""
        hechos = [
            self.modelos_listos,
            self.hay_muestra,
            self.muestra_completa,
            self.est is not None and self.est.conteos == self.conteos()
        ]
        etiquetas = [
            ("modelos", "Modelos"),
            ("imagenes", "Imágenes cargadas"),
            ("identificacion", "Identificación"),
            ("bayes", "Estimación bayesiana")] 
        out, actual_puesto = [], False
        for (clave, et), hecho in zip(etiquetas, hechos):
            if hecho:
                out.append((clave, et, "hecho"))
            elif not actual_puesto:
                out.append((clave, et, "actual"))
                actual_puesto = True
            else:
                out.append((clave, et, "bloqueado"))
        return out

  
    # ------------------------------------------------------------------
    # Etapa 3: identificacion de piezas con K-means
    # ------------------------------------------------------------------
    @staticmethod
    def leer_foto(ruta):
        try:
            gris = pre.leer_gris(ruta)
        except OSError:                 # el archivo no existe, se movio o no hay permiso
            gris = None
        if gris is None:
            raise ErrorSistema(f"No se pudo leer la imagen: {Path(ruta).name}\n"
                               "Formatos admitidos: " + ", ".join(sorted(EXT_IMAGEN))
                               + ". Las fotos HEIC del iPhone hay que pasarlas a JPG.")
        return gris

    def identificar_gris(self, gris):
        """Imagen en grises -> dict(clase, distancia, binaria, distancias={clase: d}, x, nombres)."""
        if not self.kmeans_listo:
            raise ErrorSistema("Falta cargar el modelo de imagenes (K-means).")
        try:
            binaria, _ = pre.preprocesar(gris, self.params_img)
            if binaria is None:
                raise ErrorSistema("No se detecto ninguna pieza en la foto. Usa fondo claro, buena luz y una sola pieza.")
            x = car.vector(binaria, self.nombres_car)
        except ErrorSistema:
            raise
        except Exception as e:
            raise ErrorSistema(f"No se pudo analizar la foto ({type(e).__name__}: {e}).")
        d = np.linalg.norm(self.centroides - (x - self.media) / self.desvio, axis=1)
        j = int(d.argmin())
        return dict(clase=self.clases_cluster[j], distancia=float(d[j]), binaria=binaria,
                    distancias=dict(zip(self.clases_cluster, (float(v) for v in d))),
                    x=x, nombres=list(self.nombres_car))

    def identificar_foto(self, ruta):
        """Identifica UNA foto suelta (no toca la sesion): sirve para probar con objetos reales."""
        if not self.kmeans_listo:
            raise ErrorSistema("Falta cargar el modelo de imagenes (K-means).")
        return self.identificar_gris(self.leer_foto(ruta))

    def identificar_pieza(self, i, ruta_foto=None):
        """Identifica la imagen i de las imágenes cargadas."""
        if not self.hay_muestra:
            raise ErrorSistema("Primero hay que cargar las 10 imágenes.")
        if not 0 <= i < len(self.piezas):
            raise ErrorSistema("Índice de imagen inválido.")
        if ruta_foto is None:
            ruta_foto = self.piezas[i]["ruta"]
        r = self.identificar_foto(ruta_foto)
        self.est = None
        self.piezas[i].update(
            ruta=str(ruta_foto),
            clase=r["clase"],
            distancia=r["distancia"],
            binaria=r["binaria"],
            detalle=r
        )
        self.log(
            f"Imagen {i + 1}: {Path(ruta_foto).name} -> "
            f"{r['clase']} (distancia {r['distancia']:.2f})"
        )
        return r["clase"]

    def quitar_foto(self, i):
        self.piezas[i].update(
            ruta=None, clase=None, distancia=None, binaria=None, detalle=None
        )
        self.est = None

    # ------------------------------------------------------------------
    # Etapa 4: estimacion bayesiana
    # ------------------------------------------------------------------
    def estimar(self):
        if not self.muestra_completa:
            raise ErrorSistema("Primero hay que identificar las 10 imágenes.")
        self.est = B.estimar(self.conteos())
        self.log("Estimación bayesiana calculada.")
        return self.est

    # ------------------------------------------------------------------
    # Etapa 5: comando de voz con K-NN
    # ------------------------------------------------------------------
    def reconocer_audio(self, x, fs, en_vivo=False):
        """Senal -> dict(comando | None, confianza, votos {comando: fraccion}, motivo, res).
        comando es None si no se oyo nada o si K-NN no esta seguro (menos votos que el umbral).
        en_vivo=True ademas exige que algo sobresalga del ruido de fondo (no aplica a archivos
        ya recortados)."""
        if not self.knn_listo:
            raise ErrorSistema("Falta cargar el modelo de voz (K-NN).")
        x = np.asarray(x, float)
        res, motivo = pau.preprocesar(x, int(fs), self.params_aud)
        salida = dict(comando=None, confianza=0.0, votos={}, motivo="", res=res)
        if res is None:
            salida["motivo"] = f"No se escucho ninguna palabra ({motivo})."
            return salida
        if en_vivo:
            c = contraste_db(res["filtrada"], res["fs"])
            if c < CONTRASTE_MIN_DB:
                salida["motivo"] = (f"No se escucho ninguna palabra: solo hay ruido de fondo "
                                    f"(la grabacion sobresale {c:.0f} dB y hacen falta {CONTRASTE_MIN_DB:.0f}).")
                return salida
        v = cau.extraer(res["senal"], res["fs"])
        z = ((v - self.media_a) / self.desvio_a)[None, :]
        proba = self.knn.predict_proba(z)[0]
        votos = {_sin_tildes(c): float(p) for c, p in zip(self.knn.clases_, proba)}
        cmd = _sin_tildes(self.knn.predict(z)[0])
        salida.update(votos=votos, confianza=votos[cmd])
        if votos[cmd] < UMBRAL_CONFIANZA:
            salida["motivo"] = f"No estoy seguro de la palabra (mejor opcion: {cmd}, {votos[cmd] * 100:.0f} % de los vecinos)."
        else:
            salida["comando"] = cmd
        return salida

    def reconocer_archivo(self, ruta_wav):
        try:
            x, fs = pau.leer_wav(ruta_wav)
        except (OSError, ValueError, EOFError) as e:
            raise ErrorSistema(f"No se pudo leer el audio: {e}")
        return self.reconocer_audio(x, fs, en_vivo=False)

    @staticmethod
    def grabar(segundos=2.0):
        """Graba del microfono (necesita: pip install sounddevice). Devuelve (senal, fs).
        Se graba con la frecuencia propia del microfono; el preprocesamiento la lleva a 16 kHz."""
        try:
            import sounddevice as sd
        except (ImportError, OSError) as e:
            raise ErrorSistema("Para grabar hace falta el paquete sounddevice: pip install sounddevice\n"
                               f"Mientras tanto se puede usar 'Cargar audio .wav'. ({e})")
        try:
            fs = int(sd.query_devices(kind="input")["default_samplerate"])
            x = sd.rec(int(segundos * fs), samplerate=fs, channels=1, dtype="float32")
            sd.wait()
        except Exception as e:
            raise ErrorSistema(f"No se pudo usar el microfono: {e}")
        return np.asarray(x)[:, 0].astype(float), fs

    # ------------------------------------------------------------------
    # Ejecutar un comando ya reconocido
    # ------------------------------------------------------------------
    def ejecutar(self, comando):
        """Ejecuta contar o proporcion. Salir se gestiona desde la GUI."""
        if comando == "salir":
            return True, "Cerrar la aplicación."
        permitido, motivo = self.disponible(comando)
        if not permitido:
            self.log(f"Comando '{comando}' rechazado: {motivo}")
            return False, motivo
        if comando == "contar":
            texto = B.texto_conteo(self.conteos())
        elif comando == "proporcion":
            texto = B.texto_proporcion(self.estimar())
        else:
            return False, f"Comando desconocido: {comando}"
        self.ultimo_comando = comando
        self.log(f"Comando '{comando}' ejecutado.\n{texto}")
        return True, texto