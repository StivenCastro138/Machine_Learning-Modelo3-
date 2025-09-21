# 🌳 Clasificación de Correos con Árbol de Decisión: SPAM vs HAM

Proyecto académico para la semana 5 que implementa un modelo de **Árbol de Decisión (CART)** para clasificar correos electrónicos en **SPAM** o **HAM**.

El núcleo de este proyecto no fue solo construir un clasificador, sino también realizar un análisis crítico del dataset, identificando y resolviendo un problema de **fuga de datos (data leakage)** que inicialmente conducía a una precisión irreal del 100%.

---

## 🚀 Características Principales

* **Preparación de Datos**: Limpieza de texto y vectorización con **TF-IDF**.
* **Modelo**: Implementación de `DecisionTreeClassifier` de Scikit-Learn.
* **Análisis de Data Leakage**: Identificación y eliminación de características "spoiler" (`Prioridad`, `FrecuenciaPalabrasSpam`) para construir un modelo realista.
* **Simulación Rigurosa**: Ejecución de 50 simulaciones con diferentes divisiones de datos para una evaluación robusta.
* **Métricas de Rendimiento**: Medición con **Exactitud (Accuracy)**, **F1-Score** y **Z-Score**.
* **Visualizaciones Clave**: Generación de gráficos de rendimiento y visualización del árbol de decisión final.
* **Informe Académico**: Documentación completa del proceso y los hallazgos en **LaTeX + PDF**.

---

## 📂 Estructura del Repositorio

```
📦 Clasificacion-SPAM-DecisionTree
 ┣ 📂 Dataset
 ┃ ┗ 📜 email_dataset.csv
 ┣ 📂 Graficos
 ┃ ┣ 📊 arbol_de_decision_final.png
 ┃ ┗ 📊 desempeno_final.png
 ┣ 📂 Informe
 ┃ ┣ 📂 img
 ┃ ┃ ┣ 📊 arbol_de_decision_final.png
 ┃ ┃ ┗ 📊 desempeno_final.png
 ┃ ┣ 📂 pdf
 ┃ ┃ ┗ 📜 Informe_final.pdf
 ┃ ┣ 📜 main.tex
 ┃ ┗ 📜 referencias.bib
 ┣ 📜 main.py
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```

---

## 🛠️ Requisitos

Para ejecutar este proyecto, necesitas tener Python instalado. Puedes instalar todas las dependencias necesarias ejecutando el siguiente comando en tu terminal:

```bash
pip install -r requirements.txt
```

### 📦 Librerías Principales

* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`

---

## ▶️ Ejecución

Para entrenar el modelo, ejecutar las 50 simulaciones y generar los gráficos, simplemente ejecuta el script principal desde tu terminal:

```bash
python main.py
```

Esto generará:

* Un resumen numérico del rendimiento en la consola.
* Los gráficos `arbol_de_decision_final.png` y `desempeno_final.png` en la carpeta `Graficos/`.

---

## 📑 Informe en LaTeX

El informe académico completo, que detalla la metodología, el descubrimiento de la fuga de datos y las conclusiones, está disponible en:

* 📄 **[Informe Final en PDF](Informe/pdf/Informe_final.pdf)**
* 📜 **[Código Fuente LaTeX](Informe/main.tex)**

📌 **Cómo compilar en Overleaf**:
1.  Crea un nuevo proyecto en [Overleaf](https://www.overleaf.com/).
2.  Sube los archivos `Informe_final.tex` y las imágenes de la carpeta `Graficos/`.
3.  Compila el proyecto para generar el PDF.

---

## 📈 Métricas de Rendimiento (Resultados Finales)

Después de corregir la fuga de datos, el modelo presenta un rendimiento realista y robusto. A continuación se muestra un ejemplo de los resultados obtenidos tras las 50 simulaciones:

```
==================================================
     RESUMEN DE PRECISIÓN EN LAS 50 EJECUCIONES
==================================================

Exactitud (Accuracy):
  - Precisión Promedio: 0.9580
  - Desviación Estándar: 0.0216
  - Mejor Ejecución:    1.0000
  - Peor Ejecución:     0.9000

F1-Score:
  - F1-Score Promedio:  0.9579
  - Desviación Estándar: 0.0217
```
*(Nota: Estos valores pueden variar ligeramente en cada ejecución completa del script)*

---

## 📌 Conclusiones

* El modelo de **Árbol de Decisión (CART)** es altamente efectivo para la clasificación de SPAM/HAM, logrando una precisión promedio superior al 95% con datos realistas.
* Se identificó y corrigió un severo problema de **fuga de datos**, demostrando la importancia crítica de la selección de características y el análisis exploratorio.
* La simulación de 50 ejecuciones confirmó la **estabilidad y consistencia** del modelo final.
* El proyecto cumple con todos los requisitos de la actividad, documentando de manera transparente tanto la metodología como los hallazgos.

---


## ✍️ Autor

Proyecto desarrollado como actividad académica en **Sistemas e Ingeniería**.
