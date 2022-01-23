# functions for analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm.notebook import tqdm
from gensim import models
from scipy import stats

# define function vectorize_hashtags
# to vectorize the hashtags via word2vec
def vectorize_hashtags(input_df):
    '''
    :params input_df: input dataframe with hashtags and lda output
    :return: vectorized hashtags
    '''
    # read pretrained word2vec model
    model = models.KeyedVectors.load_word2vec_format('../../data/Word2Vec/dewiki_20180420_100d.txt')

    # get lists for faster iterating
    topics_list = input_df['topic'].tolist()
    topic_words = input_df['topic_words'].tolist()
    scores = input_df['scores'].tolist()

    # retrieve vector per hashtag 
    vectors = []

    # iterate through hashtags
    for i in range(len(topics_list)):
        topic_vector = []
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
        topic_vector = np.average(word_vectors, axis=0, weights=np.array(word_scores).astype(np.float))
        vectors.append(topic_vector)

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
            
            # clean suggestions vector
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
def get_correlation(delay, hashtag_df, cluster_df, cluster_gender_df, cluster_party_df, sim_df):
    '''
    :params hashtag_df: df with hashtag time series
    :params suggestions_df: df with cluster time series
    :params sim_df: df with relevant hashtags and cluster
    :return: output dict
    '''
    output = {'hashtags':[], 'cluster':[], 'party':[], 'gender':[], 'pearsonr':[], 'p_value':[]}

    for i in tqdm(range(len(sim_df))):
        hashtag = sim_df['hashtags'][i]
        cluster = sim_df['cluster'][i]

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
        
        for party in set(cluster_party_df['party']):
            tmp = ts[ts['party']==party]
            
            tmp['cluster_count'] = tmp['cluster_count'].shift(periods=delay)
            tmp = tmp.dropna().reset_index(drop=True)
            
            # get correlation for party
            try:
                corr, p = stats.pearsonr(tmp['hashtag_count'].to_numpy(), tmp['cluster_count'].to_numpy())
            except:
                corr, p = None, None

            # append to output df
            output['hashtags'].append(hashtag)
            output['cluster'].append(cluster)
            output['party'].append(party)
            output['gender'].append('all')
            output['pearsonr'].append(corr)
            output['p_value'].append(p)

        # define time series cluster
        ts_cluster = cluster_gender_df[cluster_gender_df['cluster']==cluster][['date', 'gender', 'cluster_count']]
        ts_cluster['date'] = pd.to_datetime(ts_cluster['date'])

        # join to one df
        ts = pd.DataFrame()
        ts['date'] = pd.date_range(start='2017-05-29', end='2017-10-08')
        ts = ts.merge(ts_cluster, how='left', on='date')
        ts = ts.merge(ts_hashtag, how='left', on='date')

        for gender in set(cluster_gender_df['gender']):
            tmp = ts[ts['gender']==gender]
            
            tmp['cluster_count'] = tmp['cluster_count'].shift(periods=delay)
            tmp = tmp.dropna().reset_index(drop=True)

            # get correlation for gender
            try:
                corr, p = stats.pearsonr(tmp['hashtag_count'].to_numpy(), tmp['cluster_count'].to_numpy())
            except:
                corr, p = None, None

            # append to output df
            output['hashtags'].append(hashtag)
            output['cluster'].append(cluster)
            output['party'].append('all')
            output['gender'].append(gender)
            output['pearsonr'].append(corr)
            output['p_value'].append(p)


        # get correlation for all
        
        # define time series cluster
        ts_cluster = cluster_df[cluster_df['cluster']==cluster][['date', 'cluster_count']]
        ts_cluster['date'] = pd.to_datetime(ts_cluster['date'])
        
        # join to one df
        ts = pd.DataFrame()
        ts['date'] = pd.date_range(start='2017-05-29', end='2017-10-08')
        ts = ts.merge(ts_cluster, how='left', on='date')
        ts = ts.merge(ts_hashtag, how='left', on='date')

        ts['cluster_count'] = ts['cluster_count'].shift(periods=delay)
        ts = ts.dropna().reset_index(drop=True)
        
        try:
            corr, p = stats.pearsonr(ts['hashtag_count'].to_numpy(), ts['cluster_count'].to_numpy())
        except:
            corr, p = None, None

        # append to output df
        output['hashtags'].append(hashtag)
        output['cluster'].append(cluster)
        output['party'].append('all')
        output['gender'].append('all')
        output['pearsonr'].append(corr)
        output['p_value'].append(p)

    output = sim_df.merge(pd.DataFrame(output), how='left', on=['cluster', 'hashtags'])
    
    return output

# define function peak_analysis
def peak_analysis(test_range, sim_df, peaks_df, cluster_ts_df):
    '''
    :params range: days around peak in int
    :params sim_df: df with combinations cluster, hashtga
    :params peaks_df: df with peak days per hashtag
    :params cluster_df: df with cluster timeseries
    :return: t-test results overall and regression input detailed
    '''
    output_detail = pd.DataFrame()
    
    # get means before and after for different dims and peaks
    for hashtag in sim_df['hashtags'].unique():
        try:
            tmp = peaks_df[peaks_df['hashtag']==hashtag].reset_index(drop=True)
            cluster = sim_df[sim_df['hashtags']==hashtag]['cluster'].unique()

            for i in range(len(tmp)):
                peak_start = pd.to_datetime(tmp['peak_start'][i])
                peak_end = pd.to_datetime(tmp['peak_end'][i])
                start = peak_start - timedelta(days=test_range)
                end = peak_end + timedelta(days=test_range)
                
                for i in range(len(cluster)):
                    ts = cluster_ts_df[cluster_ts_df['cluster']==cluster[i]]
                    
                    before = ts[(ts['date']>=start)&(ts['date']<=peak_start)]
                    after = ts[(ts['date']>=peak_end)&(ts['date']<=end)]
                    
                    before_df = before.groupby(['cluster', 'party', 'gender', 'category'], as_index=False).mean()
                    before_df['time'] = 'before'

                    after_df = after.groupby(['cluster', 'party', 'gender', 'category'], as_index=False).mean()
                    after_df['time'] = 'after'

                    if before_df.empty:
                        pass
                    elif after_df.empty:
                        pass
                    else:
                        df = pd.concat([before_df, after_df])
                        df['hashtag'] = hashtag
                        df['peak'] = peak_start + timedelta(days=3)
                        df['test_range'] = test_range
                        output_detail = pd.concat([output_detail, df])

        except:
            pass

    return output_detail

# define function peak_ranges
# to extract the min and max of lda_dates per peak
def peak_ranges(x):
    mins = []
    maxs = []
    
    dates = x['lda_dates']
    n_peaks = x['num_peaks']

    for i in range(int(n_peaks)):
        mins.append(min(np.array_split(dates, n_peaks)[i].tolist()))
        maxs.append(max(np.array_split(dates, n_peaks)[i].tolist()))
    
    return pd.Series([mins,maxs],index=['peak_start','peak_end'])