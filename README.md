# Amazon Electronics — Análisis con Machine Learning

> **Grupo:** GATOBYTE · Ivonne Colque · Dilan Mamani · Tania Pérez · Ignacio Retamozo · Adriana Rocha

## Descripción del Proyecto

Este proyecto aplica técnicas de **Machine Learning** sobre un subconjunto de **1 millón de reseñas** de productos electrónicos de Amazon (extraído de un corpus de 40M) para construir tres sistemas de análisis:

| Módulo | Tarea | Variable Objetivo |
|---|---|---|
| Análisis de Sentimiento | Clasificación multiclase | Positivo / Neutro / Negativo |
| Regresión y Helpfulness | Regresión + Clasificación | Rating continuo · Votos útiles |
| Clustering | Segmentación no supervisada | Grupos de productos y reseñas |

La metodología adoptada es **CRISP-DM**, con procesamiento acelerado por GPU usando cuML/RAPIDS.

---

## Estructura del Repositorio

```
AMAZONELECTRONICS-GATOBYTE/
│
├── data/
│   └── processed/
│       └── reports/
│           ├── cleaning_metadata.png      # Gráfico: distribución post-limpieza
│           ├── report_metadata.csv        # Reporte de metadatos del dataset
│           ├── report_reviews.csv         # Reporte estadístico de reseñas
│           └── sample_class_balance.png   # Gráfico: balance de clases
│
├── notebooks/
│   ├── amazon_full_pipeline.ipynb                    # Pipeline completo de preprocesamiento
│   ├── Analisis_Sentimiento_Electronics_GATOBYTE.ipynb  # Módulo de sentimiento
│   └── CLUSTERING_GATOBYTE_AUTOML_.ipynb             # Módulo de clustering + AutoML
│
├── .gitignore
└── requirements.txt
```

---

## Dataset

El dataset utilizado es el **Amazon Electronics Reviews**, del cual se saco un subconjunto de 1 millón de registros con las siguientes características:

| Atributo | Detalle |
|---|---|
| Registros | 1,000,000 (de 40M disponibles) |
| Features | 21 columnas |
| Categorías principales | All Electronics (27.8%), Computers (26.4%), Cell Phones (10%) |
| Distribución de sentimiento | 74% Positivo · 21% Negativo · 7% Neutro |
| Compras verificadas | 100% verified_purchase |

### Variables clave

- `text` — texto de la reseña (feature principal para NLP)
- `rating` — calificación 1–5 estrellas
- `sentiment` — clase objetivo derivada del rating
- `helpful_vote` — votos de utilidad de la reseña
- `price`, `average_rating`, `rating_number` — métricas del producto
- `price_imputed` — flag de imputación de precio (control de sesgo)

> **Data Leakage Prevention:** Las variables `rating`, `is_satisfied` y `average_rating` fueron excluidas del entrenamiento ya que constituyen la base lógica del target de sentimiento.

---

## Instalación y Reproducción

### 1. Clonar el repositorio

```bash
git clone https://github.com/Alkima-hydra/AmazonElectronics-GATOBYTE.git
cd AMAZONELECTRONICS-GATOBYTE
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Nota GPU:** Los notebooks utilizan `cuML` (RAPIDS) para aceleración GPU. Si no dispones de GPU NVIDIA compatible, reemplaza los imports de `cuml` por sus equivalentes de `sklearn`.

### 3. Preparar los datos

Ejecutar primero el pipeline de preprocesamiento:

```bash
# En Jupyter o Google Colab
notebooks/amazon_full_pipeline.ipynb
```

El pipeline genera el archivo `sample_ml.parquet` requerido por los demás notebooks.

### 4. Ejecutar los módulos

```bash
# Análisis de Sentimiento
notebooks/Analisis_Sentimiento_Electronics_GATOBYTE.ipynb

# Clustering + AutoML
notebooks/CLUSTERING_GATOBYTE_AUTOML_.ipynb
```

---

## Metodología

El proyecto sigue las fases de **CRISP-DM**:

```
Business Understanding → Data Understanding → Data Preparation
        ↓
   Modeling → Evaluation → (Deployment ready)
```

### Preprocesamiento NLP (Sentimiento)

- Eliminación de stopwords y caracteres especiales
- Corrección de contracciones (`contractions`)
- Stemming con `PorterStemmer`
- Vectorización **TF-IDF** hasta 20,000 características
- Semilla global `SEED = 42` para reproducibilidad

### Feature Engineering (Clustering)

Se generaron 6 variables derivadas sobre el dataset original:

| Variable | Descripción |
|---|---|
| `helpfulness_rate` | Ratio de votos útiles sobre total de reseñas del producto |
| `rating_vs_avg` | Diferencia entre el rating de la reseña y el promedio del producto |
| `popularity_score` | `average_rating × log1p(rating_number)` |
| `review_age_days` | Antigüedad de la reseña en días |
| `is_long_review` | Binaria: reseña supera el percentil 75 de longitud |
| `main_category_clean` | Categoría consolidada (categorías pequeñas → "Otros") |

---

## Resultados Principales

### Módulo I — Análisis de Sentimiento

| Modelo | F1-Macro | Bal. Accuracy | ROC-AUC | F1-Neutro |
|---|---|---|---|---|
| Naive Bayes | 0.5438 | 0.5438 | 0.9067 | 0.06 |
| Logistic Regression | 0.6748 | 0.6748 | 0.94 | 0.30 |
| XGBoost | — | 0.7492 | — | — |
| **LightGBM ★** | **0.6922** | **0.7582** | — | **0.38** |

**Modelo ganador: LightGBM** con F1-Macro de **0.6922** (supera el umbral del 0.65). Alcanza un Recall del 97% en la clase positiva y el mejor equilibrio en las clases minoritarias.

> La métrica principal es **F1-Macro** para penalizar el sesgo hacia la clase mayoritaria (74% positivo).

---

### Módulo II — Regresión de Rating y Clasificación de Helpfulness

Este módulo aborda dos subproblemas sobre el dataset de reseñas:

- **Regresión:** predecir el rating numérico continuo (1.0 – 5.0)
- **Clasificación:** predecir si una reseña recibirá votos de utilidad (`helpful_vote`)

Se trabajó con un **30% estratificado** del dataset para eficiencia computacional, manteniendo la representatividad de las clases.

#### Regresión del Rating

| Modelo | RMSE Train | RMSE Val | MAE Val | R² Val |
|---|---|---|---|---|
| Regresión Lineal (baseline) | 0.3841 | 0.3867 | — | 0.9335 |
| LightGBM | — | — | ~0.29 | — |
| **CatBoost ★** | **Mejor** | **Mejor** | **0.29** | **~0.934+** |

**Modelo ganador: CatBoost** — seleccionado por su MAE de **0.29** y su capacidad nativa de manejar variables categóricas con regularización L2, minimizando el impacto de outliers en calificaciones extremas.

> La Regresión Lineal mostró sorprendente robustez (R² ≈ 0.9335), evidenciando una excelente preparación de los datos. CatBoost y LightGBM mejoran en los casos donde la linealidad no es suficiente.

#### Clasificación de Helpfulness (Votos Útiles)

| Modelo | AUC Val | Recall | Precisión | F1 | GAP |
|---|---|---|---|---|---|
| Logistic Reg. Baseline | 0.7443 | Bajo | Alta | — | -0.0028 |
| Logistic Reg. L2 + Balanced | 0.7456 | 0.61 | 0.43 | — | ~0 |
| Árbol de Decisión | 0.7379 | — | — | — | 0.0010 |
| **LightGBM ★** | **Mejor** | **Mejor** | **Mejor** | **Mejor** | **~0** |

**Modelo ganador: LightGBM** — La estrategia de balanceo de clases fue crítica: triplicó el Recall del baseline, pasando de detectar **2,695 a 8,690 reseñas útiles (TP)** y reduciendo drásticamente los falsos negativos.

#### Sistema Dual de Predicción

El pipeline de producción combina ambos modelos:

- **CatBoost** → predice el `rating` (escala 1.0 – 5.0)
- **LightGBM** → predice si la reseña será `útil` (0 / 1) con probabilidad asociada

---


### Módulo III — Clustering de Productos

| Modelo | Silhouette | Distribución Clusters | Observación |
|---|---|---|---|
| Baseline KMeans k=3 (sin preprocesamiento) | 0.9783   | 99.7% / 0.3% | Score inflado artificialmente |
| Mejora 1: KMeans k=3 + Pipeline | 0.2682 | 48% / 42% / 10% | Clusters reales y balanceados |
| Mejora 2: KMeans k=3 óptimo | 0.2681 | 48% / 42% / 10% | Confirma convergencia |
| Mejora 3: MiniBatchKMeans | 0.1704 | 42% / 38% / 20% | Menos cohesionado |
| **PCA + KMeans k=3 ★** | **0.311** | **Balanceado** | **Modelo ganador** |

Los 3 clusters de productos identificados:

- **Premium:** precio elevado, buena valoración, baja popularidad
- **Popular Accesible:** precio $10–$50, alta interacción, muchas reseñas
- **Nicho/Desconocido:** baja visibilidad, comportamiento atípico (~10% del catálogo)

### Clustering de Reseñas

El Silhouette Plot identificó **k=2** como número óptimo. Modelo ganador: **KMeans k=2** con Silhouette **0.4124**.

---

### Validación con AutoML

Se implementó un proceso AutoML personalizado (sklearn) evaluando **KMeans, MiniBatchKMeans y AgglomerativeClustering** sobre múltiples valores de k:

| Dominio | Combinaciones probadas | Silhouette Manual | ¿AutoML supera al manual? |
|---|---|---|---|
| Productos | 12 (3 alg × 4 k) | 0.311 |  Manual confirmado |
| Reseñas | 9 (3 alg × 3 k) | 0.412 |  Manual confirmado |

Para el módulo de sentimiento, **FLAML AutoML** (500 seg, F1-Macro optimizado) ratificó la selección de LightGBM.

> En todos los módulos, los modelos desarrollados manualmente igualaron o superaron los resultados de AutoML.

---

## Stack Tecnológico

| Librería | Uso |
|---|---|
| `cuML` / `cuDF` (RAPIDS) | Procesamiento y modelos acelerados por GPU |
| `LightGBM` | Modelo ganador sentimiento y helpfulness |
| `XGBoost`, `CatBoost` | Gradient boosting comparativo |
| `scikit-learn` | Pipelines, métricas, clustering baseline |
| `FLAML` | AutoML supervisado para sentimiento |
| `Polars` | Manipulación eficiente de datos (1M+ filas) |
| `NLTK` | Preprocesamiento NLP (stopwords, stemming) |
| `Optuna` | Optimización de hiperparámetros |
| `SHAP` | Explicabilidad de modelos |
| `Plotly` / `Matplotlib` | Visualizaciones interactivas y estáticas |

---

## Integrantes

| Nombre | GitHub |
|---|---|
| Adriana Nathalie Rocha Vedia | [@AdrianaRochaVedia](https://github.com/AdrianaRochaVedia) |
| Ivonne Micaela Colque Murillo | — |
| Dilan Obed Mamani Pamuri | — |
| Tania Morelia Pérez Dick | — |
| Ignacio Retamozo Torrez | — |



<div align="center">
  <sub>Proyecto académico · Machine Learning · 2026</sub>
</div>
