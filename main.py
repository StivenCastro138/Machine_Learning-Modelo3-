import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os

# --- 0. Preparación de Carpetas ---
if not os.path.exists('Graficos'):
    os.makedirs('Graficos')

# --- Fase 1: Preparación de Datos (Sin Fuga de Información) ---

try:
    df = pd.read_csv('Dataset/email_dataset.csv')
except FileNotFoundError:
    print("Error: Asegúrate de que el archivo 'email_dataset.csv' está en la carpeta 'Dataset'.")
    exit()

df['TextoCompleto'] = df['Asunto'].fillna('') + ' ' + df['Cuerpo'].fillna('')
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'\S+@\S+', '', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = texto.strip()
    return texto
df['TextoCompleto'] = df['TextoCompleto'].apply(limpiar_texto)

vectorizer = TfidfVectorizer(max_features=100)
X_text = vectorizer.fit_transform(df['TextoCompleto']).toarray()
X_text_features = pd.DataFrame(X_text, columns=vectorizer.get_feature_names_out())

# Eliminamos todas las columnas que puedan contener fugas de datos.
features_numericas = ['LongitudTexto', 'ProporcionMayus', 'URLs', 'NumDestinatarios']
print("Se han eliminado las columnas con posible fuga de datos para un modelo realista.")
X_numericas = df[features_numericas]

# Combinamos únicamente los metadatos crudos con el análisis de texto
X = pd.concat([
    X_numericas.reset_index(drop=True), 
    X_text_features.reset_index(drop=True)
], axis=1)

le_clase = LabelEncoder()
y = le_clase.fit_transform(df['Clase'])

print("Fase de preparación de datos completada.")

# --- Bucle de Simulación ---

accuracy_scores = []
f1_scores = []
n_runs = 50

print(f"Iniciando simulación de {n_runs} ejecuciones con el modelo final...")

for i in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i, stratify=y)
    
    # Usamos un modelo sin restricciones para ver su verdadero potencial con datos limpios
    model = DecisionTreeClassifier(random_state=i)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)

print("Simulación finalizada.")

# --- Fase 6: Gráficos y Visualizaciones ---

# 1. VISUALIZACIÓN DEL ÁRBOL DE DECISIÓN
print("\nGenerando visualización del Árbol de Decisión...")
final_model = DecisionTreeClassifier(random_state=42)
final_model.fit(X, y)

plt.figure(figsize=(25, 15))
plot_tree(final_model, 
          feature_names=X.columns.astype(str), 
          class_names=le_clase.classes_, 
          filled=True, 
          rounded=True,
          fontsize=10)
plt.title("Visualización del Árbol de Decisión (Sin Fuga de Datos)", fontsize=24)
plt.savefig("Graficos/arbol_de_decision_final.png")
plt.show()

# 2. GRÁFICO DE DESEMPEÑO
accuracy_mean = np.mean(accuracy_scores)
accuracy_std = np.std(accuracy_scores)
z_scores_accuracy = [(score - accuracy_mean) / accuracy_std if accuracy_std > 0 else 0 for score in accuracy_scores]

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(figsize=(15, 8))
ax1.set_xlabel('Número de Ejecución', fontsize=12)
ax1.set_ylabel('Valor de la Métrica (Accuracy / F1)', fontsize=12)
ax1.plot(range(n_runs), accuracy_scores, marker='o', linestyle='-', label='Exactitud (Accuracy)', alpha=0.7)
ax1.plot(range(n_runs), f1_scores, marker='x', linestyle='--', label='F1-Score', alpha=0.7)
ax1.axhline(accuracy_mean, color='blue', lw=2, linestyle=':', label=f'Promedio Exactitud: {accuracy_mean:.4f}')
ax1.set_ylim(0.8, 1.0) # Ajustamos el eje Y para ver mejor la variación
ax2 = ax1.twinx()
ax2.set_ylabel('Z-Score de la Exactitud', fontsize=12, color='red')
ax2.plot(range(n_runs), z_scores_accuracy, marker='.', linestyle=':', label='Z-Score (Accuracy)', color='red', alpha=0.7)
fig.suptitle('Desempeño del Modelo Final (Sin Fuga de Datos)', fontsize=16)
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("Graficos/desempeno_final.png")
plt.show()

# --- Fase 7: Resumen de Precisión Numérica ---

print("\n" + "="*50)
print("     RESUMEN DE PRECISIÓN EN LAS 50 EJECUCIONES")
print("="*50)
print(f"\nExactitud (Accuracy):")
print(f"  - Precisión Promedio: {np.mean(accuracy_scores):.4f}")
print(f"  - Desviación Estándar: {np.std(accuracy_scores):.4f}")
print(f"  - Mejor Ejecución:    {np.max(accuracy_scores):.4f}")
print(f"  - Peor Ejecución:     {np.min(accuracy_scores):.4f}")

print(f"\nF1-Score:")
print(f"  - F1-Score Promedio:  {np.mean(f1_scores):.4f}")
print(f"  - Desviación Estándar: {np.std(f1_scores):.4f}")
print("\n" + "="*50)
