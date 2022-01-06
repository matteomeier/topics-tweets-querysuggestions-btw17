import pandas as pd
import numpy as np
from sklearn import cluster
from sklearn import metrics
from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm

# define function to kmeans_suggestion
# kmeans clustering with different k
# returns list of scores
def kmeans_suggestions(vectorized_suggestions, min_num_cluster, max_num_cluster):
    '''
    :params vectorized_suggestions: vectorized suggestions as list
    :params min_num_topics: min number of cluster
    :params max_num_topics: max number of cluster
    :return: dict of scores and num cluster
    '''
    
    scores = {'num_cluster':[], 'inertia':[], 'distortion':[], 'silhouette_score':[], 'calinski_harabasz_score':[]}

    # kmeans clustering for different num cluster and return coherence
    for i in tqdm(range(min_num_cluster, max_num_cluster+1)):
        kmeans = cluster.KMeans(n_clusters=i, random_state=1410)
        kmeans.fit(vectorized_suggestions)
        labels = kmeans.predict(vectorized_suggestions)

        # append to dict
        scores['num_cluster'].append(i)
        scores['inertia'].append(kmeans.inertia_)
        scores['distortion'].append(sum(np.min(cdist(vectorized_suggestions, kmeans.cluster_centers_,'euclidean'), axis=1)) / vectorized_suggestions.shape[0])
        scores['silhouette_score'].append(metrics.silhouette_score(vectorized_suggestions, labels))
        scores['calinski_harabasz_score'].append(metrics.calinski_harabasz_score(vectorized_suggestions, labels))

    return scores

# define function to dbscan_suggestions
# dbscan clustering with different parameters
# returns list of silhuoette scores
def dbscan_suggestions(vectorized_suggestions):
    '''
    :params vectorized_suggestions: vectorized suggestions as list
    :return: dict of scores and parameters
    '''
    
    scores = {'eps':[], 'min_samples':[], 'silhouette_score':[], 'num_cluster':[], 'num_noise':[]}

    # dbscan clustering for different eps and different min_samples and return silhuoette
    for i in tqdm(np.arange(0.05, 1, 0.05)):
        for j in np.arange(5, 16):
            dbscan = cluster.DBSCAN(eps=i, min_samples=j).fit(vectorized_suggestions)
            labels = dbscan.labels_

            # drop noise points from labels
            tmp = pd.DataFrame()
            tmp['labels'] = labels
            tmp['vector'] = vectorized_suggestions.tolist()
            tmp = tmp[tmp['labels']!=-1]
            labels_clean = tmp['labels'].tolist()
            vectors_clean = np.array(tmp['vector'].tolist())

            # append to dict
            scores['eps'].append(i)
            scores['min_samples'].append(j)
            scores['silhouette_score'].append(metrics.silhouette_score(vectors_clean, labels_clean))
            scores['num_cluster'].append(len(set(labels_clean)) - (1 if -1 in labels else 0))
            scores['num_noise'].append(list(labels).count(-1))
    return scores
