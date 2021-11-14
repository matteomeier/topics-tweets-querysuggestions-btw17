# functions to process twitter data
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