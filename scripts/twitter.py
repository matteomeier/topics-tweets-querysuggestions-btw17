# functions to process twitter data
import ast

# define function get hashtags
# for extracting the hashtags per tweet from dictionary
def get_hashtags(x):
    '''
    :param x: entities column from twitter df
    :return: hashtags as list
    '''
    hashtags = []
    if type(x)==dict:
        tag_dict = x.get('hashtags', {})
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