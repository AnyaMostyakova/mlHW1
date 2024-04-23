import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import os

# Функция для вычисления расстояния между двумя точками
def calculate_distance(point1, point2):
   return np.sqrt(np.sum((point1 - point2) ** 2))

# Загрузка датасета
iris_data = load_iris()
features = iris_data.data

# Определение оптимального количества кластеров с помощью библиотеки sklearn
silhouette_scores = []
for num_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)  # Явное указание параметра n_init
    cluster_labels = kmeans.fit_predict(features)
    silhouette_avg = silhouette_score(features, cluster_labels)
    silhouette_scores.append(silhouette_avg)

optimal_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

# Создание папки для сохранения изображений шагов алгоритма
os.makedirs('kmeans_steps', exist_ok=True)

# Собственная реализация алгоритма K-means
def my_kmeans(data, num_clusters, max_iterations=100):
    num_samples, num_features = data.shape
    centroids = data[np.random.choice(num_samples, num_clusters, replace=False)]
    cluster_assignments = np.zeros(num_samples)

    for step in range(max_iterations):
        distances = np.array([np.linalg.norm(data - centroid, axis=1) for centroid in centroids])
        new_cluster_assignments = np.argmin(distances, axis=0)

        if np.array_equal(cluster_assignments, new_cluster_assignments):
            break

        cluster_assignments = new_cluster_assignments.copy()

        # Визуализация шага
        plt.figure()
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown']
        for cluster in range(num_clusters):
            plt.scatter(data[cluster_assignments == cluster][:, 0], data[cluster_assignments == cluster][:, 1], label=f'Cluster {cluster+1}', c=colors[cluster])
            plt.scatter(centroids[cluster][0], centroids[cluster][1], color='black', marker='x', s=100, label='Centroid')
        plt.title(f'Step: {step+1}')
        plt.legend()
        step_image_path = os.path.join('kmeans_steps', f'step_{step+1}.png')
        plt.savefig(step_image_path)
        plt.close()

        # Обновление центроидов
        for i in range(num_clusters):
            centroids[i] = data[cluster_assignments == i].mean(axis=0)

    return centroids, cluster_assignments

# Применение алгоритма к датасету Iris с оптимальным количеством кластеров
final_centroids, final_cluster_assignments = my_kmeans(features[:, 2:4], optimal_num_clusters)
