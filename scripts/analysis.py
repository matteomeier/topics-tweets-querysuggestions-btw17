# functions for analysis
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from gensim import models
from gensim.models import Word2Vec

# define function vectorize_hashtags
# to vectorize the hashtags via word2vec
# and lda output
def vectorize_hashtags(input_df):
    '''
    :params input_df: input dataframe with hashtags and lda output
    :return: vectorized hashtags
    '''
    # read pretrained word2vec model
    model = models.KeyedVectors.load_word2vec_format('../../data/Word2Vec/dewiki_20180420_100d.txt')

    # get lists for faster iterating
    hashtags = input_df['hashtag'].tolist()
    topic_words = input_df['topic_words'].tolist()
    scores = input_df['scores'].tolist()

    # retrieve vector per hashtag 
    vectors = []

    # iterate through hashtags
    for i in range(len(hashtags)):
        hashtag_vector = []
        word_vectors = []
        
        # get word scores from lda
        word_scores = scores[i]
        topics = topic_words[i]
        for j in reversed(range(len(topics))):
            try:
                # append word vector from word2vec model
                word_vectors.append(model[topics[j]])
            except:
                # clean lists 
                topics.pop(j)
                word_scores.pop(j)
        
        # get weighted average vector per hashtag and save them in one list
        hashtag_vector = np.average(word_vectors, axis=0, weights=np.array(word_scores).astype(np.float))
        vectors.append(hashtag_vector)

    return vectors

# define function compare_vectors
# to compare the hashtag vectors and the suggesion cluster vectors
# via cosine similarity
def compare_vectors(hashtag_vectors, cluster_df):
    '''
    :params hashtag_vectors: list of hashtag vectors
    :params cluster_df: pd dataframe with cluster and their vectors
    :return: output dataframe
    '''
    # retrieve lists for suggestion and suggestions word2vec vectors
    suggestions = []
    for i in range(len(cluster_df)):
        suggestions.append(cluster_df['suggestion'][i])

    sugg_vectors = cluster_df['vector'].tolist()

    # get similarity score for every suggestion per hashtag
    sim_scores = []

    # iterate through suggestions
    for i in tqdm(range(len(suggestions))):
        tmp = []
        for vector in hashtag_vectors:
            
            # clean suggestions vector because of shitty format
            sugg_vector = sugg_vectors[i]
            
            # calculate cosine similarity for suggestions vector and hashtag vector
            score = round(np.dot(sugg_vector, vector)
                        / (np.linalg.norm(sugg_vector)
                        * np.linalg.norm(vector)),3)
            tmp.append(score)
        sim_scores.append(tmp)

    cluster = cluster_df['cluster'].tolist()

    # create output
    output = pd.DataFrame({'suggestion': suggestions, 'cluster': cluster, 'similarity_scores': sim_scores})

    return output

# define function get_correlation
# to compare the hashtag timeseries and the suggesion cluster timeseries
# via pearson correlation
def get_correlation(hashtag_df, suggestions_df, similarity_df):
    '''
    :params hashtag_df: df with hashtag time series
    :params suggestions_df: df with cluster time series
    :params similarity_df: df with relevant hashtags and cluster
    :return: output dict
    '''
    
    cluster_df = suggestions_df.groupby(['date', 'cluster'], as_index=False).sum('count')
    cluster_df.rename(columns={'count':'cluster_count'}, inplace=True)

    cluster_party_df = suggestions_df.groupby(['date', 'party', 'cluster'], as_index=False).sum('count')
    cluster_party_df.rename(columns={'count':'cluster_count'}, inplace=True)

    cluster_gender_df = suggestions_df.groupby(['date', 'gender', 'cluster'], as_index=False).sum('count')
    cluster_gender_df.rename(columns={'count':'cluster_count'}, inplace=True)

    hashtag_df.rename(columns={'count':'hashtag_count'}, inplace=True)

    output = {'hashtags':[], 'cluster':[], 'party':[], 'gender':[], 'correlation':[]}

    for i in tqdm(range(len(similarity_df))):
        hashtag = similarity_df['hashtags'][i]
        cluster = similarity_df['cluster'][i]

        # define time series hashtag
        ts_hashtag = hashtag_df[hashtag_df['hashtag']==hashtag][['date', 'hashtag_count']]
        ts_hashtag['date'] = pd.to_datetime(ts_hashtag['date'])

        # define time series cluster
        ts_cluster = cluster_party_df[cluster_party_df['cluster']==cluster][['date', 'party', 'cluster_count']]
        ts_cluster['date'] = pd.to_datetime(ts_cluster['date'])

        # join to one df
        ts = pd.DataFrame()
        ts['date'] = pd.date_range(start='2017-05-29', end='2017-10-08')
        ts = ts.merge(ts_cluster, how='left', on='date')
        ts = ts.merge(ts_hashtag, how='left', on='date')
        
        for party in set(suggestions_df['party']):
            tmp = ts[ts['party']==party]
            
            # get correlation for party
            corr = tmp['hashtag_count'].corr(tmp['cluster_count'], method='pearson')
            
            # append to output df
            output['hashtags'].append(hashtag)
            output['cluster'].append(cluster)
            output['party'].append(party)
            output['gender'].append('all')
            output['correlation'].append(corr)

        # define time series cluster
        ts_cluster = cluster_gender_df[cluster_gender_df['cluster']==cluster][['date', 'gender', 'cluster_count']]
        ts_cluster['date'] = pd.to_datetime(ts_cluster['date'])

        # join to one df
        ts = pd.DataFrame()
        ts['date'] = pd.date_range(start='2017-05-29', end='2017-10-08')
        ts = ts.merge(ts_cluster, how='left', on='date')
        ts = ts.merge(ts_hashtag, how='left', on='date')

        for gender in set(suggestions_df['gender']):
            tmp = ts[ts['gender']==gender]
            
            # get correlation for gender
            corr = tmp['hashtag_count'].corr(tmp['cluster_count'], method='pearson')
            
            # append to output df
            output['hashtags'].append(hashtag)
            output['cluster'].append(cluster)
            output['party'].append('all')
            output['gender'].append(gender)
            output['correlation'].append(corr)

        # get correlation for all
        
        # define time series cluster
        ts_cluster = cluster_df[cluster_df['cluster']==cluster][['date', 'cluster_count']]
        ts_cluster['date'] = pd.to_datetime(ts_cluster['date'])
        
        # join to one df
        ts = pd.DataFrame()
        ts['date'] = pd.date_range(start='2017-05-29', end='2017-10-08')
        ts = ts.merge(ts_cluster, how='left', on='date')
        ts = ts.merge(ts_hashtag, how='left', on='date')
        corr = ts['hashtag_count'].corr(ts['cluster_count'], method='pearson')
            
        # append to output df
        output['hashtags'].append(hashtag)
        output['cluster'].append(cluster)
        output['party'].append('all')
        output['gender'].append('all')
        output['correlation'].append(corr)

    output = similarity_df.merge(pd.DataFrame(output), how='left', on=['cluster', 'hashtags'])
    
    return output