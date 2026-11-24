"""
INTERFAZ GRAFICA del clasificador de piezas con vision artificial controlado por voz.

    python gui.py

Une todas las etapas del trabajo. Tiene tres pestanas:

  Operación         el flujo completo del sistema:
                    1. cargar 10 imágenes
                    2. identificar las piezas con K-means
                    3. calcular la estimación bayesiana
                    4. ejecutar "proporcion" o "contar" por voz
                    Salir cierra la GUI en cualquier momento.
  Identificar foto  clasifica la foto de UNA pieza (por ejemplo sacada con el celular).
  Entrenamiento     entrena el K-means de imagenes y el K-NN de voz con las bases de datos.

Las reglas de que se puede hacer y cuando estan en agente.py (clase Agente); aca solo se muestra.
Para grabar la voz hace falta:  pip install sounddevice   (sin eso se puede cargar un .wav).
"""

import os
import queue
import shlex
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from tkinter import font as tkfont

from PIL import Image, ImageOps, ImageTk

import agente as A
import graficos as G

BASE = Path(__file__).resolve().parent

COLOR_HECHO, COLOR_ACTUAL, COLOR_BLOQUEADO = "#2e8b57", "#2a6fbb", "#d9dde3"
COLOR_OK, COLOR_MAL, COLOR_AVISO, COLOR_NEUTRO = "#1e7e34", "#c0392b", "#b9770e", "#444444"
NOMBRE_COMANDO = {"proporcion": "PROPORCIÓN", "contar": "CONTAR", "salir": "SALIR"}
EXTENSIONES = " ".join("*" + e for e in sorted(A.EXT_IMAGEN))
TIPOS_IMAGEN = [("Imágenes", EXTENSIONES), ("Todos los archivos", "*.*")]
TIPOS_AUDIO = [("Audio WAV", "*.wav"), ("Todos los archivos", "*.*")]


# ----------------------------------------------------------------------
# Imagenes
# ----------------------------------------------------------------------
def abrir_pil(ruta):
    """Abre una foto para mostrarla (respeta la orientacion del celular). None si no se puede."""
    try:
        return ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")
    except Exception:
        return None


def a_foto(img, ancho, alto):
    """PIL.Image -> imagen de tkinter, ajustada sin deformarla a ancho x alto."""
    img = img.copy()
    img.thumbnail((max(int(ancho), 1), max(int(alto), 1)))
    return ImageTk.PhotoImage(img)


# ----------------------------------------------------------------------
# Contenedor con barras de desplazamiento (para pantallas chicas)
# ----------------------------------------------------------------------
class Desplazable(ttk.Frame):
    """Marco que se desplaza si el contenido no entra en la ventana y se estira si sobra lugar.
    Los widgets se agregan a self.interior."""

    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)
        self.interior = ttk.Frame(self.canvas)
        self._ventana = self.canvas.create_window((0, 0), window=self.interior, anchor="nw")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.interior.bind("<Configure>", self._al_cambiar_contenido)
        self.canvas.bind("<Configure>", self._al_cambiar_ventana)
        self.bind("<Enter>", self._activar_rueda)
        self.bind("<Leave>", self._desactivar_rueda)

    def _al_cambiar_contenido(self, _evento=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _al_cambiar_ventana(self, evento):
        self.canvas.itemconfigure(self._ventana,
                                  width=max(evento.width, self.interior.winfo_reqwidth()),
                                  height=max(evento.height, self.interior.winfo_reqheight()))

    def _activar_rueda(self, _evento=None):
        self.canvas.bind_all("<MouseWheel>", self._rueda)

    def _desactivar_rueda(self, _evento=None):
        self.canvas.unbind_all("<MouseWheel>")

    def _rueda(self, evento):
        # La lista de piezas y los cuadros de texto tienen su propio desplazamiento
        if evento.widget.winfo_class() in ("Treeview", "Text"):
            return
        self.canvas.yview_scroll(int(-evento.delta / 120), "units")


# ----------------------------------------------------------------------
# Aplicacion
# ----------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Clasificador de piezas por visión artificial controlado por voz")
        self.esc = max(root.winfo_fpixels("1i") / 96.0, 1.0)      # factor de escala de la pantalla
        self.agente = A.Agente(base=BASE)
        self.cola = queue.Queue()               # resultados de los hilos de trabajo
        self.ocupado = False
        self.registro_mostrado = 0
        self.sel = None                         # pieza seleccionada en la lista
        self.imagen_entrenamiento = None

        self.configurar_estilo()
        self.ajustar_ventana()
        self.construir()
        root.protocol("WM_DELETE_WINDOW", self.cerrar)

        self.agente.iniciar_registro()
        avisos = self.agente.cargar_modelos()
        self.refrescar()
        if avisos:
            self.aviso("Faltan modelos: entrenarlos en la pestaña Entrenamiento.", COLOR_AVISO)
            self.nb.select(self.tab_ent)
        else:
            self.aviso("Listo. Sortear una caja para empezar.", COLOR_NEUTRO)
        root.after(80, self.sondear)

    # ------------------------------------------------------------------
    # Aspecto
    # ------------------------------------------------------------------
    def configurar_estilo(self):
        for nombre in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            tkfont.nametofont(nombre).configure(size=10)
        self.fuente_grande = tkfont.nametofont("TkDefaultFont").copy()
        self.fuente_grande.configure(size=15, weight="bold")
        self.fuente_media = tkfont.nametofont("TkDefaultFont").copy()
        self.fuente_media.configure(size=11, weight="bold")
        ttk.Style().configure("Grande.TButton", font=self.fuente_media, padding=8)

    def ajustar_ventana(self):
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = min(int(1260 * self.esc), sw - 40), min(int(740 * self.esc), sh - 100)
        self.root.geometry(f"{w}x{h}+{max((sw - w) // 2, 0)}+{max((sh - h) // 3, 0)}")
        self.root.minsize(min(900, w), min(520, h))

    # ------------------------------------------------------------------
    # Construccion de la ventana
    # ------------------------------------------------------------------
    def construir(self):
        # Barra de estado (abajo) primero, para que la pestanas ocupen el resto
        barra = ttk.Frame(self.root)
        barra.pack(side="bottom", fill="x")
        self.lbl_estado = tk.Label(barra, text="", anchor="w", font=self.fuente_media, fg=COLOR_NEUTRO)
        self.lbl_estado.pack(side="left", fill="x", expand=True, padx=8, pady=4)
        self.progreso = ttk.Progressbar(barra, mode="indeterminate", length=140)
        self.progreso.pack(side="right", padx=8)

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True)
        self.tab_op, self.tab_foto, self.tab_ent = Desplazable(self.nb), Desplazable(self.nb), Desplazable(self.nb)
        self.nb.add(self.tab_op, text="  Operación  ")
        self.nb.add(self.tab_foto, text="  Identificar foto  ")
        self.nb.add(self.tab_ent, text="  Entrenamiento  ")
        self.construir_operacion(self.tab_op.interior)
        self.construir_foto(self.tab_foto.interior)
        self.construir_entrenamiento(self.tab_ent.interior)

    # --- pestana Operacion --------------------------------------------------
    def construir_operacion(self, p):
        p.columnconfigure(0, weight=3)
        p.columnconfigure(1, weight=2)
        p.columnconfigure(2, weight=3)
        p.rowconfigure(1, weight=1)

        # Encabezado: pasos y modelos
        enc = ttk.Frame(p)
        enc.grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(8, 2))
        self.chips_pasos = []
        for i in range(5):
            lbl = tk.Label(enc, text="", padx=10, pady=3, font=self.fuente_media)
            lbl.pack(side="left", padx=(0, 4))
            self.chips_pasos.append(lbl)
        derecha = ttk.Frame(enc)
        derecha.pack(side="right")
        self.lbl_modelos = ttk.Label(derecha, text="")
        self.lbl_modelos.pack(side="left", padx=8)
        self.btn_recargar = ttk.Button(derecha, text="Recargar modelos", command=self.on_recargar)
        self.btn_recargar.pack(side="left")

        # Columna 1: carga e identificación de imágenes
        f1 = ttk.LabelFrame(p, text=" 1-3 · Imágenes y estimación ")
        f1.grid(row=1, column=0, sticky="nsew", padx=8, pady=6)

        fila = ttk.Frame(f1)
        fila.pack(fill="x", padx=6, pady=(6, 2))
        self.btn_varias = ttk.Button(
            fila,
            text="Cargar 10 imágenes…",
            command=self.on_varias
        )
        self.btn_varias.pack(side="left", padx=2)
        columnas = (
            ("n", "N°", 35),
            ("foto", "Imagen", 130),
            ("clase", "Identificada como", 125),
            ("dist", "Dist.", 45),
        )
        self.tree = ttk.Treeview(
            f1,
            columns=[c[0] for c in columnas],
            show="headings",
            height=10,
            selectmode="browse"
        )
        for clave, titulo, ancho in columnas:
            self.tree.heading(clave, text=titulo)
            self.tree.column(
                clave,
                width=ancho,
                anchor="w" if clave in ("foto", "clase") else "center"
            )
        self.tree.pack(fill="x", padx=8, pady=4)
        self.tree.bind("<<TreeviewSelect>>", self.on_seleccion)
        prev = ttk.Frame(f1)
        prev.pack(fill="x", padx=6, pady=6)
        self.lbl_prev_foto = ttk.Label(
            prev, text="foto", anchor="center", relief="groove"
        )
        self.lbl_prev_foto.pack(
            side="left", expand=True, fill="both", padx=2
        )
        self.lbl_prev_bin = ttk.Label(
            prev, text="imagen binaria", anchor="center", relief="groove"
        )
        self.lbl_prev_bin.pack(
            side="left", expand=True, fill="both", padx=2
        )
        
        # Columna 2: voz
        f2 = ttk.LabelFrame(p, text=" 4 · Comando de voz (K-NN) ")
        f2.grid(row=1, column=1, sticky="nsew", padx=4, pady=6)
        fila = ttk.Frame(f2)
        fila.pack(fill="x", padx=8, pady=(8, 2))
        self.btn_grabar = ttk.Button(fila, text="Grabar comando", style="Grande.TButton", command=self.on_grabar)
        self.btn_grabar.pack(side="left")
        ttk.Label(fila, text=" Duración (s):").pack(side="left", padx=(10, 2))
        self.var_seg = tk.DoubleVar(value=2.0)
        ttk.Spinbox(fila, from_=1.0, to=4.0, increment=0.5, textvariable=self.var_seg, width=5).pack(side="left")
        self.btn_wav = ttk.Button(f2, text="Cargar audio .wav…", command=self.on_wav)
        self.btn_wav.pack(anchor="w", padx=8, pady=2)
        self.lbl_voz = tk.Label(f2, text="Esperando un comando", font=self.fuente_media, fg=COLOR_NEUTRO,
                                wraplength=int(330 * self.esc), justify="left", anchor="w")
        self.lbl_voz.pack(fill="x", padx=8, pady=4)

        fila = ttk.Frame(f2)
        fila.pack(fill="x", padx=8, pady=2)
        self.chips_cmd, self.barras_voto, self.lbl_voto = {}, {}, {}
        for cmd in A.COMANDOS:
            chip = tk.Label(fila, text=NOMBRE_COMANDO[cmd], padx=8, pady=2, font=self.fuente_media)
            chip.pack(side="left", padx=(0, 4))
            self.chips_cmd[cmd] = chip
        votos = ttk.Frame(f2)
        votos.pack(fill="x", padx=8, pady=4)
        for fila_n, cmd in enumerate(A.COMANDOS):
            ttk.Label(votos, text=NOMBRE_COMANDO[cmd].capitalize(), width=11).grid(row=fila_n, column=0, sticky="w")
            barra = ttk.Progressbar(votos, orient="horizontal", mode="determinate", maximum=100, length=150)
            barra.grid(row=fila_n, column=1, padx=4, pady=1)
            etiqueta = ttk.Label(votos, text="0 %", width=6)
            etiqueta.grid(row=fila_n, column=2)
            self.barras_voto[cmd], self.lbl_voto[cmd] = barra, etiqueta
        ttk.Label(f2, text="Votos de los vecinos más cercanos").pack(anchor="w", padx=8)

        self.lbl_audio = ttk.Label(f2, text="gráfico de la grabación", anchor="center", relief="groove")
        self.lbl_audio.pack(fill="both", expand=True, padx=8, pady=6)

        self.var_respaldo = tk.BooleanVar(value=False)
        ttk.Checkbutton(f2, text="Respaldo sin micrófono: ejecutar con botones", variable=self.var_respaldo,
                        command=self.refrescar).pack(anchor="w", padx=8)
        fila = ttk.Frame(f2)
        fila.pack(fill="x", padx=8, pady=(2, 8))
        self.btn_manual = {}
        for cmd in ("proporcion", "contar"):
            b = ttk.Button(
                fila,
                text=NOMBRE_COMANDO[cmd].capitalize(),
                command=lambda c=cmd: self.on_manual(c)
            )
            b.pack(side="left", padx=2)
            self.btn_manual[cmd] = b
        self.btn_salir = ttk.Button(
            fila,
            text="Salir",
            command=self.cerrar
        )
        self.btn_salir.pack(side="left", padx=2)

        # Columna 3: resultado y registro
        f3 = ttk.LabelFrame(p, text=" Resultado del comando ")
        f3.grid(row=1, column=2, sticky="nsew", padx=8, pady=6)
        self.lbl_res_img = ttk.Label(f3, text="Acá aparece el resultado de\n«Contar» o «Proporción»", anchor="center",
                                     relief="groove")
        self.lbl_res_img.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        pestanas = ttk.Notebook(f3)
        pestanas.pack(fill="x", padx=8, pady=(0, 8))
        self.txt_resultado = scrolledtext.ScrolledText(pestanas, height=8, width=44, wrap="word", state="disabled")
        self.txt_registro = scrolledtext.ScrolledText(pestanas, height=8, width=44, wrap="word", state="disabled")
        pestanas.add(self.txt_resultado, text="Texto")
        pestanas.add(self.txt_registro, text="Registro del agente")

    # --- pestana Identificar foto ------------------------------------------
    def construir_foto(self, p):
        p.columnconfigure(0, weight=1)
        p.columnconfigure(1, weight=0)
        fila = ttk.Frame(p)
        fila.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        self.btn_ident = ttk.Button(fila, text="Cargar foto de una pieza…", style="Grande.TButton",
                                    command=self.on_identificar)
        self.btn_ident.pack(side="left")
        ttk.Label(fila, text="  Una sola pieza, sobre oscuro y con buena luz. No modifica la sesión de la pestaña Operación."
                  ).pack(side="left")

        imgs = ttk.Frame(p)
        imgs.grid(row=1, column=0, sticky="nsew", padx=8)
        self.lbl_id_foto = ttk.Label(imgs, text="foto", anchor="center", relief="groove")
        self.lbl_id_foto.pack(side="left", fill="both", expand=True, padx=2)
        self.lbl_id_bin = ttk.Label(imgs, text="imagen binaria", anchor="center", relief="groove")
        self.lbl_id_bin.pack(side="left", fill="both", expand=True, padx=2)

        der = ttk.Frame(p)
        der.grid(row=1, column=1, sticky="n", padx=8)
        self.lbl_id_clase = tk.Label(der, text="—", font=self.fuente_grande, fg=COLOR_ACTUAL)
        self.lbl_id_clase.pack(pady=(0, 6))
        self.lbl_id_dist = ttk.Label(der, text="")
        self.lbl_id_dist.pack()
        self.lbl_id_graf = ttk.Label(der, text="")
        self.lbl_id_graf.pack(pady=6)
        self.tree_car = ttk.Treeview(der, columns=("nombre", "valor"), show="headings", height=7)
        self.tree_car.heading("nombre", text="Característica")
        self.tree_car.heading("valor", text="Valor")
        self.tree_car.column("nombre", width=140)
        self.tree_car.column("valor", width=100, anchor="e")
        self.tree_car.pack()

    # --- pestana Entrenamiento ---------------------------------------------
    def construir_entrenamiento(self, p):
        p.columnconfigure(0, weight=1)
        p.columnconfigure(1, weight=1)
        self.var_dir_img = tk.StringVar(value=str(BASE / "G:\Mi unidad\Inteligencia Artificial 1"))
        self.var_dir_voz = tk.StringVar(value=str(BASE / "Audios_Entrenamiento"))
        self.var_val_img, self.var_val_voz = tk.StringVar(value="0.3"), tk.StringVar(value="0.3")
        self.var_extra_img, self.var_extra_voz = tk.StringVar(value=""), tk.StringVar(value="")
        self.var_k = tk.StringVar(value="")

        f = ttk.LabelFrame(p, text=" Imágenes: K-means ")
        f.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.campos_entrenamiento(f, self.var_dir_img, self.var_val_img, self.var_extra_img,
                                  "Carpeta con las fotos (subcarpetas: tornillos, clavos, arandelas, tuercas)",
                                  "Opciones de preprocesado (ej.: --umbral relativo --factor-relativo 0.8)")
        self.btn_ent_img = ttk.Button(f, text="Entrenar K-means", style="Grande.TButton",
                                      command=lambda: self.on_entrenar("imagenes"))
        self.btn_ent_img.pack(padx=8, pady=8, anchor="w")

        f = ttk.LabelFrame(p, text=" Voz: K-NN ")
        f.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self.campos_entrenamiento(f, self.var_dir_voz, self.var_val_voz, self.var_extra_voz,
                                  "Carpeta con los audios (subcarpetas: proporcion, contar, salir)",
                                  "Opciones de preprocesado (ej.: --umbral-db 30)")
        fila = ttk.Frame(f)
        fila.pack(fill="x", padx=8, pady=2)
        ttk.Label(fila, text="K de K-NN (vacío = automático):").pack(side="left")
        ttk.Entry(fila, textvariable=self.var_k, width=5).pack(side="left", padx=4)
        self.btn_ent_voz = ttk.Button(f, text="Entrenar K-NN", style="Grande.TButton",
                                      command=lambda: self.on_entrenar("voz"))
        self.btn_ent_voz.pack(padx=8, pady=8, anchor="w")

        fila = ttk.Frame(p)
        fila.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8)
        ttk.Label(fila, text="Ver gráfico:").pack(side="left")
        for texto, ruta in (("PCA imágenes", "resultados/clusters_2d.png"),
                            ("PCA imágenes con nombres", "resultados/clusters_2d_nombres.png"),
                            ("PCA voz", "resultados_audio/pca_audio.png"),
                            ("Elección de K", "resultados_audio/knn_eleccion_k.png")):
            ttk.Button(fila, text=texto, command=lambda r=ruta: self.mostrar_entrenamiento(BASE / r)).pack(side="left", padx=3)

        self.lbl_ent_img = ttk.Label(p, text="Acá se ven los gráficos del entrenamiento", anchor="center", relief="groove")
        self.lbl_ent_img.grid(row=2, column=0, sticky="nsew", padx=8, pady=6)
        self.txt_consola = scrolledtext.ScrolledText(p, height=18, width=70, wrap="word", state="disabled",
                                                     font=tkfont.nametofont("TkFixedFont"))
        self.txt_consola.grid(row=2, column=1, sticky="nsew", padx=8, pady=6)
        p.rowconfigure(2, weight=1)

    def campos_entrenamiento(self, f, var_dir, var_val, var_extra, texto_dir, texto_extra):
        ttk.Label(f, text=texto_dir).pack(anchor="w", padx=8, pady=(6, 0))
        fila = ttk.Frame(f)
        fila.pack(fill="x", padx=8)
        ttk.Entry(fila, textvariable=var_dir).pack(side="left", fill="x", expand=True)
        ttk.Button(fila, text="…", width=3, command=lambda: self.elegir_carpeta(var_dir)).pack(side="left", padx=3)
        fila = ttk.Frame(f)
        fila.pack(fill="x", padx=8, pady=2)
        ttk.Label(fila, text="Fracción reservada para validar (0 a 0.5):").pack(side="left")
        ttk.Entry(fila, textvariable=var_val, width=5).pack(side="left", padx=4)
        ttk.Label(f, text=texto_extra).pack(anchor="w", padx=8)
        ttk.Entry(f, textvariable=var_extra).pack(fill="x", padx=8, pady=2)

    # ------------------------------------------------------------------
    # Utilidades de la ventana
    # ------------------------------------------------------------------
    def banco_por_defecto(self):
        prueba = BASE / "imagenes_prueba"
        return prueba if prueba.is_dir() else BASE / "imagenes_originales"

    def habilitar(self, widget, si):
        try:
            widget.state(["!disabled"] if si else ["disabled"])
        except (AttributeError, tk.TclError):
            widget.configure(state="normal" if si else "disabled")

    def aviso(self, texto, color=COLOR_NEUTRO):
        self.lbl_estado.configure(text=texto, fg=color)

    def poner(self, label, img, ancho, alto, texto=""):
        """Muestra una imagen PIL en un Label (o el texto si no hay imagen)."""
        if img is None:
            label.configure(image="", text=texto)
            label.imagen = None
            return
        foto = a_foto(img, ancho * self.esc, alto * self.esc)
        label.configure(image=foto, text="")
        label.imagen = foto                     # hay que conservar la referencia

    def escribir(self, caja, texto, borrar=True):
        caja.configure(state="normal")
        if borrar:
            caja.delete("1.0", "end")
        caja.insert("end", texto)
        caja.see("end")
        caja.configure(state="disabled")

    def elegir_carpeta(self, var):
        carpeta = filedialog.askdirectory(initialdir=var.get() or str(BASE), title="Elegir carpeta")
        if carpeta:
            var.set(carpeta)

    def pedir_archivos(self, tipos, titulo, varios=False):
        if varios:
            res = filedialog.askopenfilenames(title=titulo, filetypes=tipos, initialdir=str(BASE))
            return list(self.root.tk.splitlist(res))
        res = filedialog.askopenfilename(title=titulo, filetypes=tipos, initialdir=str(BASE))
        return [res] if res else []

    def mostrar_error(self, e, detalle=""):
        if isinstance(e, A.ErrorSistema):
            self.aviso(str(e).splitlines()[0], COLOR_MAL)
            messagebox.showwarning("Atención", str(e))
        else:
            self.aviso(f"Error inesperado: {e}", COLOR_MAL)
            messagebox.showerror("Error inesperado", detalle or str(e))

    # ------------------------------------------------------------------
    # Tareas largas en hilos (la ventana no se congela)
    # ------------------------------------------------------------------
    def tarea(self, trabajo, al_terminar, mensaje="Trabajando…"):
        if self.ocupado:
            return
        self.ocupado = True
        self.aviso(mensaje, COLOR_ACTUAL)
        self.progreso.start(12)
        self.refrescar()

        def correr():
            try:
                resultado = trabajo()
            except Exception as e:              # se muestra en la ventana, no en el hilo
                self.cola.put(("err", e, traceback.format_exc()))
                return
            self.cola.put(("ok", resultado, al_terminar))

        threading.Thread(target=correr, daemon=True).start()

    def sondear(self):
        """Recoge lo que terminaron los hilos y actualiza la ventana (corre en el hilo principal)."""
        try:
            while True:
                item = self.cola.get_nowait()
                tipo = item[0]
                if tipo == "linea":
                    self.escribir(self.txt_consola, item[1] + "\n", borrar=False)
                    continue
                self.ocupado = False
                self.progreso.stop()
                try:
                    if tipo == "ok":
                        item[2](item[1])
                    elif tipo == "err":
                        self.mostrar_error(item[1], item[2])
                    elif tipo == "fin":
                        self.al_terminar_entrenamiento(item[1], item[2])
                except Exception as e:
                    self.mostrar_error(e, traceback.format_exc())
                self.refrescar()
        except queue.Empty:
            pass
        self.root.after(80, self.sondear)

    # ------------------------------------------------------------------
    # Estado de la pantalla (todo sale de lo que dice el agente)
    # ------------------------------------------------------------------
    def refrescar(self):
        ag, libre = self.agente, not self.ocupado
        # La carga de imágenes requiere el modelo K-means.
        puede_cargar = libre and ag.kmeans_listo
        # Bayes debe haberse calculado después de identificar las 10 imágenes.
        bayes_listo = (
            ag.muestra_completa
            and ag.est is not None
        )
        puede_operar = libre and bayes_listo
        self.habilitar(self.btn_varias, puede_cargar)
        self.habilitar(self.btn_grabar, libre and ag.knn_listo)
        self.habilitar(self.btn_wav, libre and ag.knn_listo)
        for cmd, b in self.btn_manual.items():
            self.habilitar(
                b,
                puede_operar and self.var_respaldo.get()
            )
        # Salir siempre está habilitado.
        self.habilitar(self.btn_salir, True)
        self.habilitar(self.btn_recargar, libre)
        self.habilitar(self.btn_ident, libre and ag.kmeans_listo)
        for b in (self.btn_ent_img, self.btn_ent_voz):
            self.habilitar(b, libre)
        colores = {
            "hecho": (COLOR_HECHO, "white"),
            "actual": (COLOR_ACTUAL, "white"),
            "bloqueado": (COLOR_BLOQUEADO, "#777777")
        }
        for i, (chip, (clave, etiqueta, estado)) in enumerate(
            zip(self.chips_pasos, ag.pasos())
        ):
            fondo, texto = colores[estado]
            chip.configure(
                text=("✔ " if estado == "hecho" else f"{i + 1} ") + etiqueta,
                bg=fondo,
                fg=texto
            )
        for cmd, chip in self.chips_cmd.items():
            permitido = (
                True if cmd == "salir"
                else puede_operar and ag.disponible(cmd)[0]
            )
            chip.configure(
                bg=COLOR_HECHO if permitido else COLOR_BLOQUEADO,
                fg="white" if permitido else "#777777"
            )
        marca = lambda ok: "✔ cargado" if ok else "✘ falta"
        self.lbl_modelos.configure(
            text=f"Imágenes (K-means): {marca(ag.kmeans_listo)}   ·   "
                 f"Voz (K-NN): {marca(ag.knn_listo)}"
        )
        self.actualizar_tabla()
        self.volcar_registro()

    def actualizar_tabla(self):
        piezas = self.agente.piezas
        if len(self.tree.get_children()) != len(piezas):
            self.tree.delete(*self.tree.get_children())
            for i in range(len(piezas)):
                self.tree.insert(
                    "",
                    "end",
                    iid=str(i),
                    values=(i + 1, "", "—", "")
                )
            self.sel = None
        for i, p in enumerate(piezas):
            if p["clase"] is None:
                valores = (
                    i + 1,
                    Path(p["ruta"]).name if p["ruta"] else "",
                    "—",
                    ""
                )
            else:
                valores = (
                    i + 1,
                    Path(p["ruta"]).name,
                    p["clase"],
                    f"{p['distancia']:.2f}"
                )
            self.tree.item(str(i), values=valores, tags=())

    def volcar_registro(self):
        nuevos = self.agente.registro[self.registro_mostrado:]
        if nuevos:
            self.escribir(self.txt_registro, "".join(f"[{h}] {t}\n" for h, t in nuevos), borrar=False)
            self.registro_mostrado = len(self.agente.registro)

    def seleccionar(self, i):
        if 0 <= i < len(self.agente.piezas):
            self.tree.selection_set(str(i))
            self.tree.see(str(i))
            self.sel = i
            self.mostrar_preview(i)


    def mostrar_preview(self, i):
        piezas = self.agente.piezas
        p = piezas[i] if i is not None and 0 <= i < len(piezas) else None
        if p is None or p["ruta"] is None:
            self.poner(self.lbl_prev_foto, None, 0, 0, "foto")
            self.poner(self.lbl_prev_bin, None, 0, 0, "imagen binaria")
            return
        # Mostrar la foto original, aunque todavía no haya sido procesada.
        self.poner(
            self.lbl_prev_foto,
            abrir_pil(p["ruta"]),
            200, 140,
            "foto"
        )
        # La imagen binaria solo existe después de identificar la pieza.
        if p.get("binaria") is not None:
            imagen_binaria = Image.fromarray(p["binaria"])
            self.poner(
                self.lbl_prev_bin,
            imagen_binaria,
                200, 140,
                "imagen binaria"
            )
        else:
            self.poner(
                self.lbl_prev_bin,
                None,
                0, 0,
                "Esperando identificación..."
            )


    def on_seleccion(self, _evento=None):
        sel = self.tree.selection()
        self.sel = int(sel[0]) if sel else None
        self.mostrar_preview(self.sel)

    # ------------------------------------------------------------------
    # Etapas 1 a 3: caja, muestra y fotos
    # ------------------------------------------------------------------
    def al_cargar_fotos(self, errores):
        ag = self.agente
        if errores:
            messagebox.showwarning(
                "Algunas imágenes no se pudieron identificar",
                "\n\n".join(errores)
            )
            self.aviso(
                f"{len(errores)} imagen(es) con errores.",
                COLOR_AVISO
            )
        elif ag.muestra_completa:
            try:
                ag.estimar()
                self.aviso(
                    "Las 10 imágenes fueron identificadas y Bayes fue calculado.",
                    COLOR_OK
                )
            except Exception as e:
                self.mostrar_error(e)
                return
        else:
            self.aviso(
                f"Imágenes identificadas: "
                f"{ag.n_identificadas} de {len(ag.piezas)}.",
                COLOR_ACTUAL
            )
        self.refrescar()
        if self.sel is not None:
            self.mostrar_preview(self.sel)

    def identificar_en_hilo(self, asignaciones, mensaje):
        ag = self.agente
        def trabajo():
            errores = []
            for i, ruta in asignaciones:
                try:
                    ag.identificar_pieza(i, ruta)
                except Exception as e:
                    errores.append(f"{Path(ruta).name}: {e}")
            return errores
        self.tarea(trabajo, self.al_cargar_fotos, mensaje)

    def on_varias(self):
        rutas = self.pedir_archivos(
            TIPOS_IMAGEN,
            "Seleccionar exactamente 10 imágenes",
            varios=True
        )
        if not rutas:
            return
        if len(rutas) != 10:
            messagebox.showwarning(
                "Cantidad incorrecta",
                f"Seleccionaste {len(rutas)} imágenes.\n"
                "Debés seleccionar exactamente 10."
            )
            return
        ag = self.agente
        try:
            # Preparar en el agente las 10 imágenes de esta sesión.
            ag.cargar_imagenes(rutas)
        except Exception as e:
            self.mostrar_error(e)
            return
        self.limpiar_resultados()
        self.sel = 0
        self.actualizar_tabla()
        self.seleccionar(0)
        asignaciones = [(i, ruta) for i, ruta in enumerate(rutas)]
        self.identificar_en_hilo(
            asignaciones,
            "Identificando las 10 imágenes con K-means…"
        )

    # ------------------------------------------------------------------
    # Etapa 4: comando de voz
    # ------------------------------------------------------------------
    def sonido_inicio(self):
        try:
            import winsound
            winsound.Beep(880, 120)
        except Exception:
            self.root.bell()

    def segundos(self):
        try:
            return min(max(float(self.var_seg.get()), 1.0), 4.0)
        except (tk.TclError, ValueError):
            return 2.0

    def reconocer_y_ejecutar(self, x, fs, en_vivo):
        ag = self.agente
        r = ag.reconocer_audio(x, fs, en_vivo=en_vivo)
        r["ok"], r["texto"] = None, ""
        if r["comando"]:
            cmd = r["comando"]
            ag.log(
                f"Comando de voz reconocido: {cmd} "
                f"({r['confianza'] * 100:.0f} % de los vecinos)"
            )
            if cmd == "salir":
                # Salir siempre está permitido y solo cierra la GUI.
                r["ok"] = True
                r["texto"] = "Cerrando la interfaz."
            else:
                permitido, motivo = ag.disponible(cmd)
                if permitido:
                    r["ok"], r["texto"] = ag.ejecutar(cmd)
                else:
                    r["ok"] = False
                    r["texto"] = motivo
        else:
            ag.log("Comando de voz no reconocido: " + r["motivo"])
        return r

    def on_grabar(self):
        seg, ag = self.segundos(), self.agente
        self.sonido_inicio()
        self.lbl_voz.configure(text=f"●  Grabando {seg:g} s… hablar ahora", fg=COLOR_MAL)

        def trabajo():
            x, fs = ag.grabar(seg)
            return self.reconocer_y_ejecutar(x, fs, True)
        self.tarea(trabajo, self.mostrar_voz, "Grabando…")

    def on_wav(self):
        rutas = self.pedir_archivos(TIPOS_AUDIO, "Audio con un comando")
        if not rutas:
            return
        ag = self.agente

        def trabajo():
            try:
                x, fs = A.pau.leer_wav(rutas[0])
            except (OSError, ValueError, EOFError) as e:
                raise A.ErrorSistema(f"No se pudo leer el audio: {e}")
            return self.reconocer_y_ejecutar(x, fs, False)
        self.tarea(trabajo, self.mostrar_voz, "Reconociendo el audio…")

    def on_manual(self, cmd):
        ag = self.agente

        def trabajo():
            ag.log(f"Comando '{cmd}' ejecutado con el botón de respaldo.")
            ok, texto = ag.ejecutar(cmd)
            return dict(comando=cmd, confianza=1.0, votos={}, motivo="", res=None, ok=ok, texto=texto)
        self.tarea(trabajo, self.mostrar_voz, "Ejecutando…")

    def mostrar_votos(self, votos):
        for cmd in A.COMANDOS:
            v = votos.get(cmd, 0.0) * 100
            self.barras_voto[cmd].configure(value=v)
            self.lbl_voto[cmd].configure(text=f"{v:.0f} %")

    def mostrar_voz(self, r):
        self.mostrar_votos(r["votos"])
        if r["res"] is not None:
            self.poner(self.lbl_audio, G.img_audio(r["res"], int(360 * self.esc), int(250 * self.esc)), 360, 250)
        cmd = r["comando"]
        if cmd is None:
            self.lbl_voz.configure(text=r["motivo"], fg=COLOR_AVISO)
            self.aviso("No se reconoció un comando. Repetir.", COLOR_AVISO)
            return
        if r["votos"]:
            self.lbl_voz.configure(text=f"Reconocido: {NOMBRE_COMANDO[cmd]}  ({r['confianza'] * 100:.0f} %)", fg=COLOR_OK)
        else:
            self.lbl_voz.configure(text=f"Comando por botón: {NOMBRE_COMANDO[cmd]}", fg=COLOR_NEUTRO)
        if cmd == "salir":
            self.cerrar()
            return
        self.mostrar_comando(cmd, r["ok"], r["texto"])

    def mostrar_comando(self, cmd, ok, texto):
        ag = self.agente
        self.escribir(self.txt_resultado, texto)
        if not ok:
            self.aviso(texto.splitlines()[0], COLOR_MAL)
            return
        ancho, alto = 460, 400
        img = None
        try:
            if cmd == "contar":
                img = G.img_conteo(
                    ag.conteos(),
                    int(ancho * self.esc),
                    int(alto * self.esc * 0.65)
                )
            elif cmd == "proporcion":
                img = G.img_estimacion(
                    ag.estimar(),
                    int(ancho * self.esc),
                    int(alto * self.esc)
                )
        except ValueError:
            img = None
        self.poner(
            self.lbl_res_img,
            img,
            ancho,
            alto,
            "Sin gráfico"
        )
        self.aviso(f"Comando «{cmd}» ejecutado.", COLOR_OK)

    def limpiar_resultados(self):
        self.poner(self.lbl_res_img, None, 0, 0, "Acá aparece el resultado de\n«Contar» o «Proporción»")
        self.poner(self.lbl_audio, None, 0, 0, "gráfico de la grabación")
        self.poner(self.lbl_prev_foto, None, 0, 0, "foto")
        self.poner(self.lbl_prev_bin, None, 0, 0, "imagen binaria")
        self.escribir(self.txt_resultado, "")
        self.mostrar_votos({})
        self.lbl_voz.configure(text="Esperando un comando", fg=COLOR_NEUTRO)

    # ------------------------------------------------------------------
    # Modelos y estimacion
    # ------------------------------------------------------------------
    def on_recargar(self):
        avisos = self.agente.cargar_modelos()
        self.refrescar()
        if avisos:
            messagebox.showwarning("Faltan modelos", "\n\n".join(avisos))
        else:
            self.aviso("Modelos cargados.", COLOR_OK)

    # ------------------------------------------------------------------
    # Pestana Identificar foto
    # ------------------------------------------------------------------
    def on_identificar(self):
        rutas = self.pedir_archivos(TIPOS_IMAGEN, "Foto de una pieza")
        if not rutas:
            return
        ag = self.agente
        self.tarea(lambda: (rutas[0], ag.identificar_foto(rutas[0])), self.mostrar_identificacion,
                   "Identificando la foto…")

    def mostrar_identificacion(self, resultado):
        ruta, r = resultado
        self.poner(self.lbl_id_foto, abrir_pil(ruta), 420, 340)
        self.poner(self.lbl_id_bin, Image.fromarray(r["binaria"]), 420, 340)
        self.lbl_id_clase.configure(text=r["clase"].upper())
        self.lbl_id_dist.configure(text=f"Distancia al centroide: {r['distancia']:.2f}")
        self.poner(self.lbl_id_graf, G.img_distancias(r["distancias"], int(360 * self.esc), int(190 * self.esc)), 360, 190)
        self.tree_car.delete(*self.tree_car.get_children())
        for nombre, valor in zip(r["nombres"], r["x"]):
            self.tree_car.insert("", "end", values=(nombre, f"{valor:.4g}"))
        self.aviso(f"La pieza se identificó como: {r['clase']}.", COLOR_OK)

    # ------------------------------------------------------------------
    # Pestana Entrenamiento
    # ------------------------------------------------------------------
    def mostrar_entrenamiento(self, ruta):
        img = abrir_pil(ruta) if Path(ruta).is_file() else None
        self.poner(self.lbl_ent_img, img, 620, 480, f"Todavía no existe {Path(ruta).name}. Entrenar primero.")

    def fraccion(self, var):
        try:
            v = float(var.get().replace(",", "."))
        except ValueError:
            raise A.ErrorSistema("La fracción de validación tiene que ser un número entre 0 y 0.5.")
        if not 0 <= v <= 0.5:
            raise A.ErrorSistema("La fracción de validación tiene que estar entre 0 y 0.5.")
        return v

    def argumentos_entrenamiento(self, tipo):
        """Lista de argumentos para correr main.py o entrenar_audio.py como un programa aparte."""
        if tipo == "imagenes":
            carpeta, val, extra = self.var_dir_img.get().strip(), self.fraccion(self.var_val_img), self.var_extra_img.get()
            script, salida_b, salida_r = "main.py", "imagenes_binarias", "resultados"
        else:
            carpeta, val, extra = self.var_dir_voz.get().strip(), self.fraccion(self.var_val_voz), self.var_extra_voz.get()
            script, salida_b, salida_r = "entrenar_audio.py", "audios_procesados", "resultados_audio"
        if not Path(carpeta).is_dir():
            raise A.ErrorSistema(f"No existe la carpeta: {carpeta}")
        args = [str(BASE / script), "-i", carpeta, "-b", str(BASE / salida_b), "-r", str(BASE / salida_r),
                "--validacion", str(val)]
        if tipo == "voz" and self.var_k.get().strip():
            if not self.var_k.get().strip().isdigit():
                raise A.ErrorSistema("K tiene que ser un número entero (o dejarlo vacío).")
            args += ["-k", self.var_k.get().strip()]
        return args + shlex.split(extra, posix=(os.name != "nt"))

    def on_entrenar(self, tipo):
        try:
            args = self.argumentos_entrenamiento(tipo)
        except A.ErrorSistema as e:
            self.mostrar_error(e)
            return
        if self.ocupado:
            return
        self.ocupado = True
        self.aviso("Entrenando… puede tardar un momento.", COLOR_ACTUAL)
        self.progreso.start(12)
        self.escribir(self.txt_consola, "> python " + " ".join(os.path.basename(a) if i == 0 else a
                                                               for i, a in enumerate(args)) + "\n\n")
        self.refrescar()
        entorno = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")

        def correr():
            try:
                proc = subprocess.Popen([sys.executable, "-u"] + args, cwd=str(BASE), stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                        env=entorno, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                for linea in proc.stdout:
                    self.cola.put(("linea", linea.rstrip("\n")))
                codigo = proc.wait()
            except Exception as e:
                self.cola.put(("err", e, traceback.format_exc()))
                return
            self.cola.put(("fin", codigo, tipo))
        threading.Thread(target=correr, daemon=True).start()

    def al_terminar_entrenamiento(self, codigo, tipo):
        if codigo != 0:
            self.aviso("El entrenamiento terminó con errores. Ver el texto de la derecha.", COLOR_MAL)
            return
        avisos = self.agente.cargar_modelos()
        self.aviso("Entrenamiento terminado y modelo cargado." if not avisos else
                   "Entrenamiento terminado. Falta entrenar el otro modelo.", COLOR_OK if not avisos else COLOR_AVISO)
        self.mostrar_entrenamiento(BASE / ("resultados/clusters_2d.png" if tipo == "imagenes"
                                            else "resultados_audio/pca_audio.png"))

    # ------------------------------------------------------------------
    def cerrar(self):
        self.root.destroy()


def main():
    if sys.platform == "win32":
        try:                                    # texto nitido en pantallas con escala 125 % / 150 %
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()