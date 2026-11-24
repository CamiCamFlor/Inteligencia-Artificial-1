from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visual general
plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

# Carpetas de entrada y salida (ajustá las rutas según tu proyecto)
carpeta_datos = Path("resultados_audio")       # Carpeta donde están los CSV
carpeta_salida = Path("graficos_audio")   # Carpeta donde se guardarán las imágenes
carpeta_salida.mkdir(parents=True, exist_ok=True)

# 1. Cargar y unir datasets ('audio' en predicciones <-> 'archivo' en caracteristicas)
pred = pd.read_csv(carpeta_datos / "predicciones_audio.csv")
carac = pd.read_csv(carpeta_datos / "caracteristicas_audio.csv")
df = pd.merge(
    pred,
    carac.drop(columns=["clase"], errors="ignore"),
    left_on="audio",
    right_on="archivo",
    how="inner",
)

classes = ["contar", "proporcion", "salir"]
palette = {
    "contar": "#2b7bba",
    "proporcion": "#d95f02",
    "salir": "#1b9e77",
}

# 2. Agregar las 160 columnas (10 segmentos x 16 descriptores) en sus 16 promedios globales por audio
feature_cols = [c for c in carac.columns if c not in ["archivo", "clase"]]
base_types = sorted(list(set("_".join(c.split("_")[1:]) for c in feature_cols)))

for bt in base_types:
    cols_bt = [f"seg{i:02d}_{bt}" for i in range(1, 11)]
    df[bt] = df[cols_bt].mean(axis=1)

main_feats = ["zcr_prom", "amp_prom", "mfcc1_prom", "mfcc2_prom", "mfcc4_prom", "mfcc5_prom"]

# 3. Métricas de clasificación y estadísticas en consola
df["acierto"] = df["comando_real"] == df["comando_predicho"]
print(f"Exactitud global: {df['acierto'].mean():.2%} ({df['acierto'].sum()}/{len(df)})")
for (split, metodo), grp in df.groupby(["conjunto", "como_se_predijo"]):
    print(f"Exactitud [{split} - {metodo}]: {grp['acierto'].mean():.2%} ({grp['acierto'].sum()}/{len(grp)})")

print("\nMatriz de confusión global (real vs predicho):")
print(pd.crosstab(df["comando_real"], df["comando_predicho"], margins=True))

print("\nResumen estadístico (media y std) de los 16 descriptores por comando:")
resumen = df.groupby("comando_real")[base_types].agg(["mean", "std"]).reindex(classes)
print(resumen.round(4))

# 4. Gráfico 1: Barras de perfil acústico (escala original y normalizado [0, 1])
means_all = df.groupby("comando_real")[base_types].mean().reindex(classes)
norm_all = (means_all - means_all.min()) / (means_all.max() - means_all.min())

fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))

means_all[main_feats].T.plot(kind="bar", ax=axes[0], color=[palette[c] for c in classes], width=0.75)
axes[0].set_title("Promedio global por descriptor y comando (escala original)")
axes[0].set_xlabel("Descriptor acústico (promedio de 10 segmentos)")
axes[0].set_ylabel("Valor promedio")
axes[0].tick_params(axis="x", rotation=25)
axes[0].legend(title="Comando real")

norm_all.T.plot(kind="bar", ax=axes[1], color=[palette[c] for c in classes], width=0.8)
axes[1].set_title("Perfil relativo normalizado [0–1] de los 16 descriptores")
axes[1].set_xlabel("Descriptor acústico")
axes[1].set_ylabel("Escala relativa (mín = 0, máx = 1)")
axes[1].tick_params(axis="x", rotation=45)
axes[1].legend(title="Comando real")

plt.tight_layout()
plt.savefig(carpeta_salida / "01_audio_perfil_caracteristicas.png", bbox_inches="tight")
plt.close()

# 5. Gráfico 2: Boxplots + puntos individuales de los 6 descriptores principales
fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
axes = axes.flatten()
for idx, feat in enumerate(main_feats):
    sns.boxplot(
        data=df, x="comando_real", y=feat, order=classes,
        palette=palette, ax=axes[idx], width=0.48, fliersize=0
    )
    sns.stripplot(
        data=df, x="comando_real", y=feat, order=classes,
        color="black", alpha=0.65, size=4.8, ax=axes[idx]
    )
    axes[idx].set_title(f"Distribución de {feat}")
    axes[idx].set_xlabel("")
    axes[idx].tick_params(axis="x", rotation=15)

plt.suptitle("Dispersión por comando de voz (promedio de 10 segmentos)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(carpeta_salida / "02_audio_distribucion_descriptores.png", bbox_inches="tight")
plt.close()

# 6. Gráfico 3: Evolución temporal por segmento (seg01 a seg10)
seg_metrics = [
    ("zcr_prom", "Tasa de Cruce por Cero (ZCR prom)"),
    ("amp_prom", "Amplitud promedio"),
    ("mfcc1_prom", "MFCC 1 promedio"),
    ("mfcc4_prom", "MFCC 4 promedio"),
]

fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.2))
axes = axes.flatten()
segs = [f"seg{i:02d}" for i in range(1, 11)]

for idx, (metric, label) in enumerate(seg_metrics):
    cols = [f"{s}_{metric}" for s in segs]
    for cmd in classes:
        sub = df[df["comando_real"] == cmd][cols]
        mean_curve = sub.mean().values
        std_curve = sub.std().values
        x = np.arange(1, 11)
        axes[idx].plot(x, mean_curve, marker="o", label=cmd, color=palette[cmd], linewidth=2)
        axes[idx].fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=palette[cmd], alpha=0.14)
    axes[idx].set_title(f"Evolución temporal: {label}")
    axes[idx].set_xlabel("Segmento temporal (1 a 10)")
    axes[idx].set_ylabel(metric)
    axes[idx].set_xticks(range(1, 11))
    axes[idx].legend(title="Comando")

plt.tight_layout()
plt.savefig(carpeta_salida / "03_audio_evolucion_segmentos.png", bbox_inches="tight")
plt.close()

# 7. Gráfico 4: Particiones y método de predicción (Leave-One-Out vs Modelo)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

split_counts = (
    pd.crosstab(df["comando_real"], df["conjunto"])
    .reindex(classes)[["entrenamiento", "validacion"]]
)
split_counts.plot(kind="bar", stacked=True, ax=axes[0], color=["#4c78a8", "#f58518"], width=0.52)
axes[0].set_title("Muestras por comando y partición (100% acierto)")
axes[0].set_xlabel("Comando real")
axes[0].set_ylabel("Cantidad de audios")
axes[0].tick_params(axis="x", rotation=15)
axes[0].legend(title="Conjunto")

eval_counts = pd.crosstab(df["comando_real"], df["como_se_predijo"]).reindex(classes)
eval_counts.plot(kind="bar", stacked=True, ax=axes[1], color=["#54a24b", "#e45756"], width=0.52)
axes[1].set_title("Esquema de evaluación por comando")
axes[1].set_xlabel("Comando real")
axes[1].set_ylabel("Cantidad de audios")
axes[1].tick_params(axis="x", rotation=15)
axes[1].legend(title="Método de predicción")

plt.tight_layout()
plt.savefig(carpeta_salida / "04_audio_particiones_evaluacion.png", bbox_inches="tight")
plt.close()

print(f"\nGráficos guardados en: {carpeta_salida.resolve()}")