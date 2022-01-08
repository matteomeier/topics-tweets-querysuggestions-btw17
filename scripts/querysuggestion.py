# functions to process querysuggestion
import pandas as pd
import numpy as np
import glob
from tqdm.notebook import tqdm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import ast
import json
from gensim import models
from gensim.models import Word2Vec


# define function concat_suggestions
# to filter dataset and concat to one df
def concat_suggestions(input_files, start, end):
    '''
    :params input_files: file list for twitter chunks
    :params start: start date of recording twitter data
    :params end: two weeks after end of recording twitter data + one day bc of datetime
    :return: pandas df
    '''
    colnames = ['id', 'queryterm', 'date', 'client', 'lang', 'url', 'raw_data']
    output = pd.DataFrame()

    # concatenate dataframes
    for index in tqdm(range(len(input_files))):
        file = input_files[index]
        
        # read df
        df = pd.read_csv(file, names=colnames, header=None)
        
        # filter df
        df = df[(df['date']>=start)&(df['date']<=end)]
        
        # get clean df
        raw_data = df['raw_data'].tolist()
        date = df['date'].tolist()

        querysuggestions_clean = []
        ranking = []
        dates = []
        queryterms = []

        for i in range(len(df)):
            querysuggestions = json.loads(raw_data[i])[1]
            queryterm = json.loads(raw_data[i])[0].casefold()
            dates.append(date[i])
            queryterms.append(queryterm)
            tmp1 = []
            tmp2 = []
            for sugg in querysuggestions:
                sugg_clean = sugg.replace(queryterm+' ', '')        
                tmp1.append(sugg_clean)
                tmp2.append(querysuggestions.index(sugg)+1)
            querysuggestions_clean.append(tmp1)
            ranking.append(tmp2)
            
        df_clean = pd.DataFrame({'date': dates, 'queryterm': queryterms, 'ranking': ranking, 'suggestion': querysuggestions_clean})
        df_clean = df_clean.set_index(['date', 'queryterm']).apply(pd.Series.explode).reset_index()
        
        output = output.append(df_clean)

    return output

# define function vectorize_suggestions
# to vectorize the querysuggestion via word2vec
def vectorize_suggestions(input_df):
    '''
    :params input_df: input dataframe with querysuggestions
    :return: suggestions and vectorized suggestions
    '''
    # read pretrained word2vec model
    model = models.KeyedVectors.load_word2vec_format('../../data/Word2Vec/dewiki_20180420_100d.txt')

    # save suggestions tokenized in list
    suggestions = [list(x) for x in set(tuple(x) for x in input_df['tokens'].tolist())]

    # retrieve suggestion-vectors
    vector_data = []
    for i in tqdm(range(len(suggestions))):
        mean_vector = []
        for j in reversed(range(len(suggestions[i]))):
            try:
                mean_vector.append(model[suggestions[i][j]])
            except:
                suggestions[i].pop(j)
        vector = np.average(mean_vector, axis=0)
        vector_data.append(vector)
    vector_data = [x for x in vector_data if x.shape==(100,)]

    vector_data = np.asarray(vector_data)

    return suggestions, vector_data