import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración visual general
plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

carpeta=Path("resultados")
# 1. Cargar y unir datasets por 'imagen'
asig = pd.read_csv(carpeta / "asignaciones.csv")
carac = pd.read_csv(carpeta / "caracteristicas.csv")
df = pd.merge(asig, carac.drop(columns=["clase"], errors="ignore"), on="imagen", how="inner")

features = ["hu1", "hu2", "hu3", "hu4", "circularidad", "vertices"]
classes = ["arandelas", "clavos", "tornillos", "tuercas"]
palette = {
    "arandelas": "#2b7bba",
    "clavos": "#d95f02",
    "tornillos": "#1b9e77",
    "tuercas": "#7570b3",
}

# 2. Métricas de clasificación y distancias en consola
df["acierto"] = df["clase_real"] == df["clase_predicha"]
print(f"Exactitud global: {df['acierto'].mean():.2%} ({df['acierto'].sum()}/{len(df)})")
for split, grp in df.groupby("conjunto"):
    print(f"Exactitud [{split}]: {grp['acierto'].mean():.2%} ({grp['acierto'].sum()}/{len(grp)})")

print("\nMatriz de confusión (real vs predicha):")
print(pd.crosstab(df["clase_real"], df["clase_predicha"], margins=True))

print("\nEstadísticas por clase real:")
resumen = df.groupby("clase_real")[features + ["distancia_al_centroide"]].agg(["mean", "std", "median"])
print(resumen.round(4))

# 3. Gráfico 1: Barras de promedios (original y normalizado [0, 1] entre clases)
means = df.groupby("clase_real")[features].mean().reindex(classes)
norm_means = (means - means.min()) / (means.max() - means.min())

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
means.T.plot(kind="bar", ax=axes[0], color=[palette[c] for c in classes], width=0.75)
axes[0].set_title("Promedio por característica y clase (escala original)")
axes[0].set_xlabel("Característica")
axes[0].set_ylabel("Valor promedio")
axes[0].tick_params(axis="x", rotation=25)
axes[0].legend(title="Clase real")

norm_means.T.plot(kind="bar", ax=axes[1], color=[palette[c] for c in classes], width=0.75)
axes[1].set_title("Perfil relativo normalizado [0–1] entre clases")
axes[1].set_xlabel("Característica")
axes[1].set_ylabel("Escala relativa (mín clase = 0, máx clase = 1)")
axes[1].tick_params(axis="x", rotation=25)
axes[1].legend(title="Clase real")

plt.tight_layout()
plt.savefig("01_perfil_caracteristicas_clases.png")  # Cambiar a .jpg si se desea
plt.close()

# 4. Gráfico 2: Boxplots + puntos por característica
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
axes = axes.flatten()
for idx, feat in enumerate(features):
    sns.boxplot(
        data=df, x="clase_real", y=feat, order=classes,
        palette=palette, ax=axes[idx], width=0.5, fliersize=0
    )
    sns.stripplot(
        data=df, x="clase_real", y=feat, order=classes,
        color="black", alpha=0.65, size=4.5, ax=axes[idx]
    )
    axes[idx].set_title(f"Distribución de {feat}")
    axes[idx].set_xlabel("")
    axes[idx].tick_params(axis="x", rotation=20)

plt.suptitle("Dispersión y casos atípicos por clase", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("02_distribucion_caracteristicas.png", bbox_inches="tight")
plt.close()

# 5. Gráfico 3: Muestras por partición y distancia al centroide
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

split_counts = (
    pd.crosstab(df["clase_real"], df["conjunto"])
    .reindex(classes)[["entrenamiento", "validacion"]]
)
split_counts.plot(kind="bar", stacked=True, ax=axes[0], color=["#4c78a8", "#f58518"], width=0.55)
axes[0].set_title("Muestras por clase y partición")
axes[0].set_xlabel("Clase real")
axes[0].set_ylabel("Cantidad de imágenes")
axes[0].tick_params(axis="x", rotation=20)
axes[0].legend(title="Conjunto")

sns.boxplot(
    data=df, x="clase_real", y="distancia_al_centroide",
    hue="conjunto", order=classes, ax=axes[1], palette=["#4c78a8", "#f58518"]
)
axes[1].set_title("Distancia al centroide por clase y conjunto")
axes[1].set_xlabel("Clase real")
axes[1].set_ylabel("Distancia al centroide")
axes[1].tick_params(axis="x", rotation=20)
axes[1].legend(title="Conjunto")

plt.tight_layout()
plt.savefig("03_particiones_y_distancias.png")
plt.close()

print("\nGráficos guardados correctamente.")