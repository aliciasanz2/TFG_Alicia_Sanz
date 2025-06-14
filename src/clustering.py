# Selección y limpieza de las columnas de interés
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Selección, limpieza y muestreo de datos
cols = [
    'Age (Yrs)',
    'Time to Retirement',
    'Director Network Size',
    'Total Number of Quoted Boards to Date'
]
# Eliminamos filas con NaN en las columnas de interés
df = dataset[cols].dropna()

# Tomamos una muestra de 5 000 filas para mejorar el rendimiento
df_sample = df.sample(n=5000, random_state=42)

# 2. Escalado de los datos
X = StandardScaler().fit_transform(df_sample.values)

# 3. Método del codo (inercia) para k = 1…10
inertias = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

# 4. Silhouette score para k = 2…10
sil_scores = []
K_sil = range(2, 11)
for k in K_sil:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    sil_scores.append(silhouette_score(X, labels))

# 5. Determinar k óptimo por silhouette
opt_k = K_sil[sil_scores.index(max(sil_scores))]

# 6. Clustering final con k óptimo
km_opt = KMeans(n_clusters=opt_k, random_state=42)
labels_opt = km_opt.fit_predict(X)

# 7. Reducción de dimensión para visualización con PCA
X_pca = PCA(n_components=2, random_state=42).fit_transform(X)

# 8. Gráfica con 3 subplots: codo, silhouette y clusters
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 8.1. Método del codo
axes[0].plot(K, inertias, '-o')
axes[0].set_title('Método del Codo (Inercia)')
axes[0].set_xlabel('Número de clusters (k)')
axes[0].set_ylabel('Inercia')

# 8.2. Silhouette
axes[1].plot(K_sil, sil_scores, '-o', color='orange')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('Número de clusters (k)')
axes[1].set_ylabel('Silhouette')

# 8.3. Scatter de clusters en el espacio PCA
for cluster in range(opt_k):
    mask = labels_opt == cluster
    axes[2].scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        s=10,
        label=f'Cluster {cluster}'
    )
axes[2].set_title(f'Clusters PCA (k={opt_k})')
axes[2].set_xlabel('PC1')
axes[2].set_ylabel('PC2')
axes[2].legend(title='Cluster')

plt.tight_layout()
plt.show()