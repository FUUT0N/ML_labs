import pandas as pd
import wikipedia
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from itertools import chain
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage

# Массивы, содержащие в себе список статей
class ARTICLES:
    NAMES = ['Mountains', 'Cooking', 'Government']

    MOUNTAINS = ['Machapuchare', 'K2', 'Kangchenjunga', 'Lhotse',
                 'Makalu', 'Cho Oyu', 'Dhaulagiri', 'Manaslu',
                 'Nanga Parbat', 'Annapurna', 'Broad Peak',
                 'Shishapangma', 'Nuptse', 'Kamet', 'Trivor']

    COOKING = ['Cooking', 'Cookbook', 'Cooker', 'Cuisine',
               'Culinary arts', 'Culinary profession', 'Food industry',
               'Nutrition', 'Recipe', 'Scented water', 'Carryover cooking',
               'List of ovens', 'Dishwashing', 'Food writing', 'Foodpairing']

    GOVERNMENT = ['Central government', 'Constitutional economics', 'Comparative government',
                  'Digital democracy', 'E-Government', 'Legal rights', 'List of countries by system of government',
                  'Political economy', 'Politics', 'Voting system', 'World government',
                  'Constitutional economics', 'Oligarchy', 'Autocracy', 'Anarchism']

# Создаём массив статей, содержащий в себе текст, и массив с названиями статей
def create_wiki_list(articles: list):
    w_list = []
    w_title = []
    for article in chain.from_iterable(articles):
        print(f'Loading content: {article}')
        w_list.append(wikipedia.page(article).content)
        w_title.append(article)
    print("Examine content")
    # Создание .pkl файлов для хранения данных статей и их названий
    pickle.dump(w_list, open("data.pkl", "wb"))
    pickle.dump(w_title, open("names.pkl", "wb"))
    return w_list, w_title

# Извлекаем фичи из текстов с помощью TF-IDF (на английском языке)
def get_tf_idf_vector(w_list: list):
    return TfidfVectorizer(stop_words={'english'}).fit_transform(w_list)

# Используем "Метод локтя" для определения числа кластеров
def elbow_method(start: int, end: int, tfidf_vec):
    # Сумма квадратов расстояния выборок до ближайших центров кластеров
    sum_squared_distances = []
    K = range(start, end)
    for k in K:
        # Метод к-средних
        km = KMeans(n_clusters=k, max_iter=300, n_init=15, random_state=0)
        km = km.fit(tfidf_vec)
        sum_squared_distances.append(km.inertia_)
    # Построение графика
    plt.plot(K, sum_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Squared distances')
    plt.title('Elbow Method For Optimal k')
    plt.savefig('elbow_method.png')
    plt.close()

# Кластеризация методом к-средних
def kmeans_clustering(k: int, tfidf_vec, title):
    model = KMeans(n_clusters=k, max_iter=200, n_init=15, random_state=0).fit(tfidf_vec)
    labels = model.labels_
    wiki_cl = pd.DataFrame(list(zip(title, labels)), columns=['Article', 'Cluster'])
    return wiki_cl

# Кластеризация методом Mini Batch K-means
def mini_batch_kmeans_clustering(k: int, tfidf_vec, title):
    model = MiniBatchKMeans(n_clusters=k, batch_size=10, max_iter=200, random_state=0).fit(tfidf_vec)
    labels = model.labels_
    wiki_cl = pd.DataFrame(list(zip(title, labels)), columns=['Article', 'Cluster'])
    return wiki_cl

# Кластеризация методом DBSCAN
def dbscan_clustering(tfidf_vec, title):
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(tfidf_vec)
    distances, indices = nbrs.kneighbors(tfidf_vec)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.show()
    plt.close()

    model = DBSCAN(eps=0.78, min_samples=2).fit(tfidf_vec)

    labels = model.labels_
    wiki_cl = pd.DataFrame(list(zip(title, labels)), columns=['Article', 'Cluster'])
    return wiki_cl

# Иерархичный метод кластеризации и построение дендрограммы
# method = 'ward' использует алгоритм минимизации дисперсии Уорда
def hierarchy_clustering(tfidf_vec, title):
    linkage_matrix = linkage(tfidf_vec.toarray(), 'ward')
    plt.figure(figsize=(15, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    dendrogram(
        linkage_matrix,
        orientation="right",
        labels=title
    )
    plt.savefig('hierarchy.png')


if __name__ == "__main__":
    # Инициализация массивов
    wiki_list = list()
    title_list = list()
    if os.path.exists("data.pkl" and "names.pkl"):
        wiki_list = pickle.load(open("data.pkl", 'rb'))
        title_list = pickle.load(open("names.pkl", 'rb'))
    else:
        wiki_list, title_list = create_wiki_list([ARTICLES.MOUNTAINS,
                                                  ARTICLES.COOKING,
                                                  ARTICLES.GOVERNMENT])

# Извлечение фич с помощью TF-IDF
    tf_idf_vector = get_tf_idf_vector(wiki_list)

# Применение "метода локтя" и построение графика
    elbow_method(2, 6, tf_idf_vector)

    n_clusters = 3

# Применение метода к-средних
    kmeans_result = kmeans_clustering(n_clusters, tf_idf_vector, title_list)
    print('KMeans method')
    print(kmeans_result)
    print()

# Применение Mini Batch K-means
    mini_batch_kmeans_result = mini_batch_kmeans_clustering(n_clusters, tf_idf_vector, title_list)
    print('MiniBatchKMeans method')
    print(mini_batch_kmeans_result)
    print()

# Применение метода DBSCAN
    dbscan_result = dbscan_clustering(tf_idf_vector, title_list)
    print('DBSCAN method')
    print(dbscan_result)

# Применение метода иерархической кластеризации и построение дендрограммы
    hierarchy_clustering(tf_idf_vector, title_list)
