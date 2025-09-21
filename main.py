print("✅ Script final iniciado. Generando todos los análisis para el informe...")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# --- CARGAR Y PREPARAR DATOS ---
try:
    df = pd.read_csv('email_dataset.csv')
except FileNotFoundError:
    print("Error: Asegúrate de que tu archivo CSV ('email_dataset.csv') esté en la misma carpeta.")
    exit()

# Mapeamos etiquetas
etiqueta_map = {'spam': 1, 'ham': 0}
df['Clase'] = df['Clase'].map(etiqueta_map)
df.dropna(subset=['Clase'], inplace=True)
df['Clase'] = df['Clase'].astype(int)

# 🔹 Eliminamos soplones directos
df = df.drop(columns=['FrecuenciaPalabrasSpam', 'ErroresOrtograficos', 'Cuerpo'], errors='ignore')

# 🔹 Creamos features
df['Asunto_Longitud'] = df['Asunto'].astype(str).apply(len)
df['Asunto_NumExclamaciones'] = df['Asunto'].astype(str).apply(lambda x: x.count('!'))
df['Asunto_NumMayus'] = df['Asunto'].astype(str).apply(lambda x: sum(1 for c in x if c.isupper()))

# 🔹 Extraemos la hora de la fecha
df['Hora'] = pd.to_datetime(df['FechaHora']).dt.hour

# 🔹 Definimos features finales
features_finales = [
    'LongitudTexto',
    'ProporcionMayus',
    'URLs',
    'NumDestinatarios',
    'Asunto_Longitud',
    'Asunto_NumExclamaciones',
    'Asunto_NumMayus',
    'Hora',
    'Formato',
    'Sector',
    'Prioridad',
    'Adjuntos'
]

# Reaplicamos encoding solo a las categóricas necesarias
df_encoded = pd.get_dummies(df[features_finales + ['Clase']], 
                            columns=['Formato', 'Sector', 'Prioridad', 'Adjuntos'], 
                            drop_first=True)

# --- ANÁLISIS DE CORRELACIÓN ---
print("\nGenerando análisis de correlación...")
numeric_features = [
    'LongitudTexto', 'ProporcionMayus', 'URLs', 
    'NumDestinatarios', 'Asunto_Longitud', 
    'Asunto_NumExclamaciones', 'Asunto_NumMayus', 'Hora'
]
correlation_matrix = df[numeric_features + ['Clase']].corr()

# Gráfico 1: Matriz de Correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Gráfico 1: Matriz de Correlación de Features Numéricas')
plt.savefig('grafico_1_correlacion.png')
print("Gráfico 1 guardado como 'grafico_1_correlacion.png'")

# --- SELECCIÓN DE CARACTERÍSTICAS Y DIVISIÓN DE DATOS ---
X = df_encoded.drop(columns=['Clase'])
y = df_encoded['Clase']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- CONSTRUCCIÓN Y ENTRENAMIENTO DEL MODELO LOGÍSTICO ---
print("\nConstruyendo y entrenando el modelo de Regresión Logística...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)
print("¡Modelo entrenado!")

# --- PREDICCIONES Y MÉTRICAS DE ERROR ---
y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

# Métricas de error y rendimiento
accuracy = (y_pred == y_test).mean()
error_rate = 1 - accuracy
report = classification_report(y_test, y_pred, output_dict=True)
precision_spam = report['1']['precision']
f1_spam = report['1']['f1-score']

print("\n--- Métricas de Rendimiento y Error ---")
print(f"Exactitud (Accuracy): {accuracy:.4f}")
print(f"Tasa de Error: {error_rate:.4f}")
print(f"Precisión para SPAM: {precision_spam:.4f}")
print(f"F1-Score para SPAM: {f1_spam:.4f}")
print("---------------------------------------")

# --- ANÁLISIS DETALLADO Y GRÁFICOS ---

# Gráfico 2: Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
title = f'Gráfico 2: Matriz de Confusión (sobre {len(y_test)} muestras de prueba)'
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['HAM', 'SPAM'], yticklabels=['HAM', 'SPAM'])
plt.xlabel('Predicción del Modelo')
plt.ylabel('Valor Real')
plt.title(title)
plt.savefig('grafico_2_matriz_confusion.png')
print("Gráfico 2 guardado como 'grafico_2_matriz_confusion.png'")

# Gráfico 3: Importancia de Características
importances = pd.DataFrame(data={'Feature': X.columns, 'Importance': model.coef_[0]})
importances = importances.sort_values(by='Importance', key=abs, ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importances.head(15))
plt.title('Gráfico 3: Importancia de cada Característica (Feature Importance)')
plt.savefig('grafico_3_importancia_features.png')
print("Gráfico 3 guardado como 'grafico_3_importancia_features.png'")

# Gráfico 4: Distribución de Probabilidades y Umbrales
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_prob)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds_roc[optimal_idx]

plt.figure(figsize=(12, 7))
sns.histplot(x=y_pred_prob, hue=y_test, kde=True, bins=50)
plt.title('Gráfico 4: Distribución de Probabilidades y Umbrales')
plt.xlabel('Probabilidad Predicha de ser SPAM')
plt.ylabel('Cantidad de Correos')
plt.axvline(0.5, color='red', linestyle='--', label=f'Umbral por Defecto (0.5)')
plt.axvline(optimal_threshold, color='green', linestyle='--', label=f'Umbral Óptimo ({optimal_threshold:.2f})')
plt.legend()
plt.savefig('grafico_4_distribucion_probabilidades.png')
print("Gráfico 4 guardado como 'grafico_4_distribucion_probabilidades.png'")

# --- VALIDACIÓN CRUZADA ---
print("\nEjecutando validación cruzada (5 folds)...")
scores_f1 = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='f1')
scores_acc = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='accuracy')

print("\n--- Validación Cruzada ---")
print(f"F1 promedio: {scores_f1.mean():.4f} ± {scores_f1.std():.4f}")
print(f"Accuracy promedio: {scores_acc.mean():.4f} ± {scores_acc.std():.4f}")
print("--------------------------")

print("\n✅ ¡Proceso completado! Revisa los 4 archivos .png y los resultados en consola.")
