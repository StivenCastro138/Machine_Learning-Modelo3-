# 📧 Clasificación de Correos: SPAM vs HAM

Proyecto académico que implementa un modelo de **Regresión Logística** para clasificar correos electrónicos en **SPAM** o **HAM**.
Incluye dataset, código en Python, métricas de rendimiento, gráficos interpretativos y un **informe técnico en LaTeX/Overleaf**.

---

## 🚀 Características principales

* Preprocesamiento del dataset (`Dataset/email_dataset.csv`).
* Ingeniería de características: remitente, asunto, longitud, proporción de mayúsculas, URLs, adjuntos, entre otros.
* Entrenamiento con **Regresión Logística**.
* Evaluación con métricas (Accuracy, Error Rate, Precision, F1).
* Validación cruzada.
* Visualizaciones clave.
* Informe académico en **LaTeX + PDF**.

---

## 📂 Estructura del repositorio

```
📦 Clasificacion-SPAM-HAM
 ┣ 📂 Dataset
 ┃ ┗ 📜 email_dataset.csv
 ┣ 📂 Gráficos
 ┃ ┣ 📊 grafico_1_correlacion.png
 ┃ ┣ 📊 grafico_2_matriz_confusion.png
 ┃ ┣ 📊 grafico_3_importancia_features.png
 ┃ ┗ 📊 grafico_4_distribucion_probabilidades.png
 ┣ 📂 Informe
 ┃ ┣ 📂 pdf
 ┃ ┃ ┗ 📜 Informe_final.pdf
 ┃ ┣ 📜 main.tex
 ┣ 📜 main.py
 ┣ 📜 requirements.txt
 ┣ 📜 README.md
```

---

## 🛠️ Requisitos

Instalar dependencias con:

```bash
pip install -r requirements.txt
```

### 📦 Librerías principales

* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`

---

## ▶️ Ejecución

Entrenar y evaluar el modelo:

```bash
python main.py
```

Esto generará:

* Métricas en consola.
* Gráficos en la carpeta `Gráficos/`.

---

## 📊 Resultados y Gráficos

### 🔹 1. Correlación de variables
### 🔹 2. Matriz de confusión
### 🔹 3. Importancia de las características
### 🔹 4. Distribución de probabilidades

---

## 📑 Informe en LaTeX

El informe académico completo está disponible en:

* 📄 [Informe Final en PDF](Informe/pdf/Informe_final.pdf)
* 📜 [Código LaTeX](Informe/main.tex)

📌 **Cómo usar en Overleaf**:

1. Descarga la carpeta `Informe/`.
2. Súbela a [Overleaf](https://www.overleaf.com/).
3. Compila con **pdfLaTeX** para generar el documento.

---

## 📈 Métricas de rendimiento

```
--- Métricas de Rendimiento y Error ---
Exactitud (Accuracy): 1.0000
Tasa de Error: 0.0000
Precisión para SPAM: 1.0000
F1-Score para SPAM: 1.0000
---------------------------------------
--- Validación Cruzada ---
F1 promedio: 1.0000 +- 0.0000
Accuracy promedio: 1.0000 +- 0.0000
```

⚠️ Estos resultados reflejan un **sobreajuste**, ya que en contextos reales el rendimiento nunca es perfecto. Esto abre la discusión sobre la necesidad de datasets más variados y representativos.

---

## 📌 Conclusiones

* La **Regresión Logística** es efectiva para tareas de clasificación binaria como SPAM vs HAM.
* El dataset empleado permitió un **100% de rendimiento**, pero se identificó riesgo de sobreajuste.
* Se evidenció la importancia de la ingeniería de características y el análisis gráfico.
* El informe académico documenta tanto la metodología como los resultados.

---

## ✍️ Autor

Proyecto desarrollado como actividad académica en **Sistemas e Ingeniería**.
