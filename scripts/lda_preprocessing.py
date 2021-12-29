# functions for lda preprocessing
import pandas as pd
import glob
from tqdm.notebook import tqdm
from datetime import datetime, timedelta
import spacy
nlp = spacy.load('de_dep_news_trf')
from nltk.corpus import stopwords
stopwords = stopwords.words('german')

# define function mark_peaks
# to mark the peak dates for identified peaks
def mark_peaks(hashtag_df, peak_df):
    '''
    :params hashtag_df: input hashtag dataframe
    :params peak_df: input peak dataframe
    :return: df with marked indices
    '''
    # mark peaks in dataframe and save to df
    hashtag_list = hashtag_df['hashtag'].unique().tolist()

    df = pd.DataFrame()

    for i in tqdm(range(len(hashtag_list))):
        hashtag = hashtag_list[i]
        small_df = hashtag_df[hashtag_df['hashtag']==hashtag].reset_index()
        small_peak_df = peak_df[peak_df['hashtag']==hashtag].reset_index()
        small_df['peak'] = 0
        for peak in small_peak_df['peak'].tolist():
            small_df.loc[int(peak), 'peak'] = 1
        df = df.append(small_df)
        
    return df

# define function retrieve_peak_dates
# to retrieve the peak dates +-3 days for identified peaks
def retrieve_peak_dates(hashtag_df, input_df):
    '''
    :params hashtag_df: input hashtag dataframe
    :params input: input dataframe
    :return: indices of peaks per hashtag
    '''
    # mark peaks in dataframe and save to df
    hashtag_list = hashtag_df['hashtag'].unique().tolist()
    lda_dates = []

    for i in tqdm(range(len(hashtag_list))):
        hashtag = hashtag_list[i]
        peak_dates = input_df[(input_df['peak']==1)&(input_df['hashtag']==hashtag)]['date'].tolist()
        for peak in peak_dates:
            start = str(datetime.strptime(str(peak), '%Y-%m-%d %H:%M:%S').date() - timedelta(days=3))
            end = str(datetime.strptime(str(peak), '%Y-%m-%d %H:%M:%S').date() + timedelta(days=3))
            daterange = pd.date_range(start, end)
            output_dates = []
            for date in daterange:
                output_dates.append(date.strftime('%Y-%m-%d'))
        if peak_dates:
            lda_dates.append(output_dates)
        else:
            lda_dates.append(peak_dates)

    output = pd.DataFrame({'hashtag': hashtag_list, 'lda_dates': lda_dates})
    return output

# define function nlp_pipeline
# to run texts from hashtags through nlp pipeline
def nlp_pipeline(text):
    '''
    :params text: text to preprocess
    :return: preprocessed text
    '''
    doc = nlp(text)
    lemma_list = [str(tok.lemma_).lower() for tok in doc
                  if tok.is_alpha and tok.text.lower() not in stopwords and len(tok.text.lower())>=3]
    return lemma_list