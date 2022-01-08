# functions for lda
import pandas as pd
from tqdm.notebook import tqdm
import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel

# define function create_documents
# to join different tweets from single user with one hashtag
# to single document and save in list
def create_documents(input_df):
    '''
    :params input_df: input dataframe
    :return: list
    '''
    hashtag_list = input_df['tags'].unique().tolist()
    documents = []

    for index in tqdm(range(len(hashtag_list))):
        tmp = input_df[input_df['tags']==hashtag_list[index]]
        user_list = tmp['username'].unique().tolist()
        for i in range(len(user_list)):
            doc = []
            tmp_user = tmp[tmp['username']==user_list[i]]
            for sublist in tmp_user['tokens']:
                for item in sublist:
                    doc.append(item)
            documents.append(doc)
    
    return documents

# define function to run lda
# topic modelling with different num topics
# return list of ldas and list of coherence scores
def run_ldas(documents, min_num_topics, max_num_topics):
    '''
    :params documents: documents as list
    :params min_num_topics: min number of topics to extract
    :params max_num_topics: max number of topics to extract
    :return: list of gensim lda models, list of coherence scores matching these models with num topics
    '''
    model_list = []
    coherence_scores = {'num_topics':[], 'c_v':[]}

    # get dictionary
    dictionary = gensim.corpora.Dictionary(documents)

    # get bag of words corpus
    bow_corpus = [dictionary.doc2bow(doc) for doc in documents]

    # run tf-idf-model and extract corpus for lda
    tfidf = gensim.models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    # run lda models for different num_topics and extract coherence scores
    for i in range(min_num_topics, max_num_topics+1):
        lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=i, id2word=dictionary, passes=2, workers=4, random_state=1410)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=documents, dictionary=dictionary, coherence='c_v')  
        coherence_lda = coherence_model_lda.get_coherence()     
        model_list.append(lda_model)
        coherence_scores['num_topics'].append(i)
        coherence_scores['c_v'].append(coherence_lda)
    
    return model_list, coherence_scores

# define function get_topicwords_scores
# to extract the topic words and their probability scores
# from lda model
def get_topicwords_scores(lda_model):
    '''
    :params lda_model: lda model to extract the information from
    :return: pd dataframe
    '''

    x = lda_model.print_topics(-1)
    topic_word_list = []
    score_list = []
    topic_list = []

    for index in range(len(x)):
        topic_words = []
        scores = []
        y = x[index][1].split('"')
        z = x[index][1].split('*')
        for i in range(len(y)):
            if i%2 != 0:
                topic_words.append(y[i])    
        for i in range(len(z)):
            if i == 0:
                scores.append(z[i])
            elif i != max(range(len(z))):
                scores.append(z[i].split('+ ')[1])         
        
        topic_word_list.append(topic_words)
        score_list.append(scores)
        topic_list.append(x[index][0])
    
    output = pd.DataFrame({'topic': topic_list, 'topic_words': topic_word_list, 'scores': score_list})
    return output

# define function get_tweet_topic
# to extract the topic for every relevant tweet
# from lda model
def get_tweet_topic(lda_model, input_df):
    '''
    :params lda_model: lda model to extract the information from
    :params input_df: input dataframe
    :return: pd dataframe
    '''
    documents = []
    for item in input_df['tokens'].tolist():
        documents.append(item)

    # get dictionary
    dictionary = gensim.corpora.Dictionary(documents)

    # get bag of words corpus
    bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
    
    # run tf-idf-model and extract corpus for lda
    tfidf = gensim.models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    # get topic for document
    topic_scores = []
    topics = []
    for i in tqdm(range(len(documents))):
        max = sorted(lda_model[corpus_tfidf[i]], key=lambda tup: -1*tup[1])[0]
        topic_scores.append(max[1])
        topics.append(max[0])

    # add columns to input_df
    input_df['topic'] = topics
    input_df['topic_score'] = topic_scores

    return input_df

# define function get_hashtag_topic
# to compute the topic for every hashtag
# from tweet topic distribution
def get_hashtag_topic(input_df, topic_df):
    '''
    :params input_df: input dataframe
    :params topic_df: topic dataframe
    :return: pd dataframe
    '''
    hashtag_list = input_df['tags'].unique()
    word_list = []
    score_list = []

    for i in tqdm(range(len(hashtag_list))):
        hashtag = hashtag_list[i]
        
        # filter input df
        tmp = input_df[input_df['tags']==hashtag]
        
        # calculate ranking score
        calc = tmp.groupby('topic', as_index=False)['topic_score'].sum()
        calc = calc.merge(topic_df, how='left', on='topic')
        calc = calc.set_index(['topic', 'topic_score']).apply(pd.Series.explode).reset_index()
        calc['word_rank_score'] = calc['topic_score'].astype('float') * calc['scores'].astype('float')
        
        # get highest ranked words
        top_words = calc.nlargest(10, columns='word_rank_score')['topic_words'].tolist()
        top_scores = calc.nlargest(10, columns='word_rank_score')['word_rank_score'].tolist()

        word_list.append(top_words)
        score_list.append(top_scores)
    
    output = pd.DataFrame({'hashtag':hashtag_list, 'topic_words':word_list, 'scores':score_list})
    return output
        

