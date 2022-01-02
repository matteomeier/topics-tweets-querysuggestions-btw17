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