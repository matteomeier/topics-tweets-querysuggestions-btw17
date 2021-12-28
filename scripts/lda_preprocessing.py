# functions for lda preprocessing
import pandas as pd
import glob
from tqdm.notebook import tqdm
from datetime import datetime, timedelta

# define function mark_peaks
# to detect the peak dates for identified peaks
def mark_peaks(input_df, peak_df):
    '''
    :params hashtag: hashtag to detect peaks
    :params input_df: input dataframe
    :return: indices of peaks per hashtag
    '''
    # mark peaks in dataframe and save to complete_df
    hashtag_list = input_df['hashtag'].unique().tolist()

    complete_df = pd.DataFrame()

    for i in tqdm(range(len(hashtag_list))):
        hashtag = hashtag_list[i]
        small_df = input_df[input_df['hashtag']==hashtag].reset_index()
        small_peak_df = peak_df[peak_df['hashtag']==hashtag].reset_index()
        small_df['peak'] = 0
        for peak in small_peak_df['peak'].tolist():
            small_df.loc[int(peak), 'peak'] = 1
        complete_df = complete_df.append(small_df)
        