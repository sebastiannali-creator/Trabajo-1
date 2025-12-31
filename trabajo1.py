import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configurar el estilo de las gráficas
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Leer el dataset
df = pd.read_csv('/Users/sebastiannandrealirodriguez/Desktop/carpeta csv.]/emotions.csv')

print("Información básica del dataset:")
print(f"Forma del dataset: {df.shape}")
print(f"Columnas: {df.columns.tolist()}")
print(f"Distribución de etiquetas:\n{df['label'].value_counts()}")
print(f"Valores nulos: {df.isnull().sum().sum()}")

# Crear figura con múltiples subplots
fig = plt.figure(figsize=(20, 15))

# 1. Distribución de etiquetas emocionales
plt.subplot(3, 3, 1)
emotion_counts = df['label'].value_counts()
colors = ['#FF6B6B', '#4ECDC4', "#338CA0"]
bars = plt.bar(emotion_counts.index, emotion_counts.values, color=colors)
plt.title('Distribución de Emociones', fontsize=14, fontweight='bold')
plt.xlabel('Tipo de Emoción')
plt.ylabel('Frecuencia')
for bar, count in zip(bars, emotion_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             str(count), ha='center', va='bottom', fontweight='bold')

# 2. Gráfica de pastel para proporciones
plt.subplot(3, 3, 2)
plt.pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
plt.title('Proporción de Emociones', fontsize=14, fontweight='bold')

# 3. Estadísticas básicas de algunas características importantes
plt.subplot(3, 3, 3)
feature_cols = [col for col in df.columns if col != 'label']
selected_features = feature_cols[:5]  # Seleccionar las primeras 5 características
df_features = df[selected_features]
boxplot_data = [df_features[col].values for col in selected_features]
bp = plt.boxplot(boxplot_data, labels=[col[:8] + '...' if len(col) > 8 else col for col in selected_features])
plt.title('Distribución de Características (Top 5)', fontsize=14, fontweight='bold')
plt.xlabel('Características')
plt.ylabel('Valores')
plt.xticks(rotation=45)

# 4. Correlación entre algunas características
plt.subplot(3, 3, 4)
correlation_features = feature_cols[:10]  # Primeras 10 características
corr_matrix = df[correlation_features].corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Matriz de Correlación (Top 10)', fontsize=14, fontweight='bold')

# 5. Distribución de medias por emoción
plt.subplot(3, 3, 5)
mean_features = [col for col in df.columns if 'mean' in col.lower()][:3]
if mean_features:
    for i, feature in enumerate(mean_features):
        for emotion in df['label'].unique():
            data = df[df['label'] == emotion][feature]
            plt.hist(data, alpha=0.5, label=f'{emotion}-{feature[:6]}', bins=20)
    plt.title('Distribución de Características "Mean"', fontsize=14, fontweight='bold')
    plt.xlabel('Valores')
    plt.ylabel('Frecuencia')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 6. Análisis PCA para visualización en 2D
plt.subplot(3, 3, 6)
# Preparar datos para PCA
X = df[feature_cols].values
y = df['label'].values

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Crear mapa de colores para cada emoción
emotion_colors = {'NEGATIVE': '#FF6B6B', 'NEUTRAL': '#4ECDC4', 'POSITIVE': '#45B7D1'}
for emotion in df['label'].unique():
    mask = y == emotion
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=emotion_colors[emotion], label=emotion, alpha=0.6, s=30)

plt.title(f'PCA - Visualización 2D\n(Varianza explicada: {pca.explained_variance_ratio_.sum():.1%})', 
          fontsize=14, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.legend()

# 7. Comparación de estadísticas por emoción
plt.subplot(3, 3, 7)
stats_by_emotion = df.groupby('label')[feature_cols[:5]].mean()
stats_by_emotion.T.plot(kind='bar', ax=plt.gca())
plt.title('Promedio de Características por Emoción', fontsize=14, fontweight='bold')
plt.xlabel('Características')
plt.ylabel('Valor Promedio')
plt.xticks(rotation=45)
plt.legend(title='Emociones')

# 8. Varianza de características por emoción
plt.subplot(3, 3, 8)
variance_by_emotion = df.groupby('label')[feature_cols[:5]].var()
variance_by_emotion.T.plot(kind='bar', ax=plt.gca())
plt.title('Varianza de Características por Emoción', fontsize=14, fontweight='bold')
plt.xlabel('Características')
plt.ylabel('Varianza')
plt.xticks(rotation=45)
plt.legend(title='Emociones')

# 9. Distribución acumulativa
plt.subplot(3, 3, 9)
selected_feature = feature_cols[0]  # Primera característica
for emotion in df['label'].unique():
    data = df[df['label'] == emotion][selected_feature]
    plt.hist(data, bins=30, alpha=0.5, label=emotion, density=True, cumulative=True)
plt.title(f'Distribución Acumulativa\n{selected_feature[:20]}...', fontsize=14, fontweight='bold')
plt.xlabel('Valores')
plt.ylabel('Densidad Acumulativa')
plt.legend()

plt.tight_layout()
plt.show()

# Imprimir información adicional sobre el PCA
print(f"\nAnálisis PCA:")
print(f"Varianza explicada por PC1: {pca.explained_variance_ratio_[0]:.3f}")
print(f"Varianza explicada por PC2: {pca.explained_variance_ratio_[1]:.3f}")
print(f"Varianza total explicada: {pca.explained_variance_ratio_.sum():.3f}")

# Estadísticas por emoción
print(f"\nEstadísticas generales por emoción:")
for emotion in df['label'].unique():
    subset = df[df['label'] == emotion]
    print(f"\n{emotion}:")
    print(f"  Número de muestras: {len(subset)}")
    print(f"  Promedio de primera característica: {subset[feature_cols[0]].mean():.3f}")
    print(f"  Desviación estándar: {subset[feature_cols[0]].std():.3f}")