# functions to process twitter data
import ast
import pandas as pd
from tqdm.notebook import tqdm
import plotly.express as px

# define function count_hashtags
# to preprocess all json input twtiter files
# and count their hashtags per day
def count_hashtags(input_files):
    '''
    :params input_files: file list for twitter chunks
    :return: pandas df
    '''
    output = pd.DataFrame()

    # iterate through files
    for index in tqdm(range(len(input_files))):
        file = input_files[index]
        with open(file, 'r') as f:
            
            # get chunk name
            chunk = file.split('chunk-')[1].split('.')[0]
            
            # read df
            df = pd.read_json(f)

            # apply function get hashtags
            df['tags'] = df['entities'].apply(get_hashtags)
            
            # explode tags to rows and drop na values
            df = df.explode('tags')
            df = df[df['tags'].notna()]
            
            # change datetime to date and lower all hashtags
            df['created_at'] = pd.to_datetime(df['created_at']).dt.date
            df['tags'] = df['tags'].apply(lambda x: str(x).lower())
            
            # aggregate dfs and groupby date and hashtag
            df_agg = pd.DataFrame()
            df_agg[['date', 'hashtag', 'count']] = df.groupby(['created_at', 'tags'], as_index=False)['id'].count()
            
            # save to output dataframe
            output = output.append(df_agg)

    output = output.groupby(['date', 'hashtag'], as_index=False).sum('count')
    return output

# define function get hashtags
# for extracting the hashtags per tweet from dictionary
def get_hashtags(entities):
    '''
    :param entities: entities column from twitter df
    :return: hashtags as list
    '''
    hashtags = []
    if type(entities)==dict:
        tag_dict = entities.get('hashtags', {})
        for item in tag_dict:
            tag = item.get('text', {})
            hashtags.append(tag)
    return hashtags

# define function get text
# for extracting full tweet text per tweet
# define function to retrieve text
def get_text(ext_tweet, retweet_stat, text):
    '''
    :param ext_tweet: column extended_tweet of twitter df
    :param retweet_stat: columns retweeted_status of twitter_df
    :param text: column text of twitter_df
    :return: full tweet text
    '''
    if ext_tweet == 0:
        if retweet_stat == 0:
            full_text = text
        else:
            full_text = retweet_stat.get('text')
    else:
        full_text = ext_tweet.get('full_text')
    return full_text

# define function plot_top_hashtags
# to plot the peak detection
# for evaluation and reporting
def plot_top_hashtags(input_df, num_hashtags):
    '''
    :params input_df: input dataframe
    :params num_hashtags: number of hashtags to plot 
    :return: plot
    '''
    # plot top 5 hashtags
    topn = input_df[['hashtag','count']].groupby('hashtag', as_index=False).sum('count').nlargest(columns='count', n=num_hashtags)
    df_plot = input_df[input_df['hashtag'].isin(topn['hashtag'])]
    df_plot.rename(columns={'hashtag':'Hashtag', 'count':'Häufigkeit', 'date':'Datum'}, inplace=True)

    fig = px.line(df_plot, x='Datum', y='Häufigkeit', color='Hashtag',
              template='simple_white', color_discrete_sequence=px.colors.qualitative.Antique)
    fig.show()

# define function filter_tweets
# to filter tweets on peak dates
def filter_tweets(input_files, dates_df):
    '''
    :params input_files: file list for twitter chunks
    :params dates_df: dataframe with hashtags and peak dates
    :return: output dataframe
    '''
    hashtag_list = dates_df['hashtag'].tolist()
    lda_dates_list = dates_df['lda_dates'].tolist()    

    created_at_list = []
    id_list = []
    text_list = []
    user_list = []
    extended_tweet_list = []
    retweeted_status_list = []
    tags_list = []

    # iterate through files create output csv
    for index in tqdm(range(len(input_files))):
        file = input_files[index]
        with open(file, 'r') as f:
            
            # read df
            df = pd.read_json(f)
            # apply function get hashtags
            df['tags'] = df['entities'].apply(get_hashtags)
            
            # explode tags to rows and drop na values
            df = df.explode('tags')
            df = df[df['tags'].notna()]
            
            # change datetime to date and lower all hashtags
            df['created_at'] = pd.to_datetime(df['created_at']).dt.date
            df['created_at'] = df['created_at'].apply(str)
            df['tags'] = df['tags'].apply(lambda x: str(x).lower())
                    
            for i in range(len(dates_df)):
                hashtag = hashtag_list[i]
                lda_dates = lda_dates_list[i]
                temp_df = df[(df['tags']==hashtag)&(df['created_at'].isin(lda_dates))]
                
                if len(temp_df)==0:
                    pass
                else:
                    # append data to lists
                    created_at_list.append(temp_df['created_at'].tolist())
                    id_list.append(temp_df['id'].tolist())
                    text_list.append(temp_df['text'].tolist())
                    user_list.append(temp_df['user'].tolist())
                    extended_tweet_list.append(temp_df['extended_tweet'].tolist())
                    retweeted_status_list.append(temp_df['retweeted_status'].tolist())
                    tags_list.append(temp_df['tags'].tolist())

    # create df
    output = pd.DataFrame(data={'created_at': created_at_list, 'id': id_list, 'text': text_list,
                                'user': user_list, 'extended_tweet': extended_tweet_list,
                                'retweeted_status': retweeted_status_list, 'tags': tags_list})

    # explode df after setting fake column
    output['A'] = 1
    output = output.set_index(['A']).apply(pd.Series.explode).reset_index()
    output.drop(columns='A', inplace=True)

    return output