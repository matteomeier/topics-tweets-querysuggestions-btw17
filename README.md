# political-topics-in-query-suggestions

This is the repository for my Bachelor-Thesis **Topic-Analyse politischer Tweets und Suchvorschläge zur Bundestagswahl 2017** in Data and Information Science at the University of Applied Sciences Cologne.
#### Structure of this repository
* `notebooks\`: .ipynb notebooks to work with, analyze and visualize data
* `reports\`: .html version of the notebooks
* `scripts\`: .py scripts for functions used in the notebooks
#### Notebook description
| filename                    | description                                                                                                                                                         |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `btw17_querysuggestions_data_structure.ipynb` | First analysis of the BTW17 query suggestions dataset. Simple matching of query suggestions. Plot used for Exposé. |
| `btw17_twitter_wavelet_preprocess.ipynb`   | Retrieve hashtag and their mentions per day for peak detection. |
| `btw17_twitter_wavelet.ipynb`   | Identify peaks per hashtag per Wavelet Transform and Kolmogorov-Zurbenko-Filter. |
| `btw17_twitter_lda_preprocess.ipynb` | Retrieve the peak date +- 3 days per hashtag for LDA. |
| `btw17_twitter_lda_preprocess_text.ipynb` | Retrieve the tweets for identified days for LDA. |
| `btw17_twitter_lda.ipynb` | Latent Dirichlet Allocation on identified peaks per hashtag. |
| 'btw17_compare_topics_suggestions.ipynb' | Word2Vec, K-Means-Clustering of topics, comparing vectors of topics and suggestions. |
