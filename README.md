# Topics in Tweets and Query Suggestions (BTW 17)

This is the repository for my Bachelor-Thesis **Topic-Analyse politischer Tweets und Suchvorschläge zur Bundestagswahl 2017** in Data and Information Science at the University of Applied Sciences Cologne.
#### Structure of this repository
* `data\`: data generated in the notebooks
* `figs\`: figures generated in the notebooks and used in thesis
* `notebooks\`: .ipynb notebooks to work with, analyze and visualize data
* `reports\`: .html version of the notebooks
* `scripts\`: .py scripts for functions used in the notebooks

#### Description of folders and relevant data files
Please note that due to late addition of data to this repository, the folders and subfolders for the data files differ from those used in the notebooks.
The file names remain unchanged, some files were not uploaded due to restrictions or redundancy.

| folder/filename                    | description                                                                                                                                                         |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `analysis` | Files created in notebook `btw17_analysis.ipynb` to answer the research questions. |
| `querysuggestions\cluster\cluster.json` | Initial cluster of query suggestions generated with DBSCAN. |
| `querysuggestions\cluster\cluster_categories.csv` | File to manually match cluster of query suggestions to topic categories. |
| `twitter\hashtags` | Hashtag time series and vector representations. |
| `twitter\lda` | Initial LDA topics, topic-hashtag-distribution and preprocessed lda tweets with topic. |
| `twitter\peaks` | Detected peak dates for hashtag time series. |

#### Notebook description
| filename                    | description                                                                                                                                                         |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `additional_descriptions_and_plots.ipynb` | Additional plots used in thesis. |
| `btw17_analysis.ipynb` | Visual analysis of the suggestions and hashtag matching. |
| `btw17_compare_topics_suggestions.ipynb` | Get similarity score for suggestions and hashtag topics. |
| `btw17_querysuggestions_data_structure.ipynb` | First analysis of the BTW17 query suggestions dataset. Simple matching of query suggestions. Plot used for Exposé. |
| `btw17_querysuggestions_dbscan.ipynb` | Retrieve relevant query suggestions (time filter) from BTW17 dataset. DBSCAN algorithm for clustering the query suggestions. |
| `btw17_twitter_lda.ipynb` | Latent Dirichlet Allocation on identified peaks per hashtag. |
| `btw17_twitter_lda_preprocess.ipynb` | Retrieve the peak date +- 3 days and the tweets per hashtag for LDA. |
| `btw17_twitter_wavelet.ipynb`   | Retrieve hashtag and their mentions per day for peak detection. Identify peaks per hashtag per Wavelet Transform and Kolmogorov-Zurbenko-Filter. |
| `plotting_wavelets.ipynb` | Plotting wavelets for thesis. |

#### Recreating the Analysis
For recreating the results of this analysis you need two datasets:
* ESuPol Query Suggestions Dataset from the University of Applied Sciences Köln
* Twitter Dataset from the University of Applied Sciences Lübeck, DOI: https://doi.org/10.3390/data2040034

Further you need the pretrained word-embeddings `dewiki_20180420_100d.txt` by Wikipedia2Vec: https://wikipedia2vec.github.io/wikipedia2vec/pretrained/

With these two datasets and the pretrained word-embeddings, execute the described notebooks in the following order:
* `btw17_twitter_wavelet.ipynb`
* `btw17_twitter_lda_preprocess.ipynb`
* `btw17_twitter_lda.ipynb`
* `btw17_querysuggestions_dbscan.ipynb`
* `btw17_compare_topics_suggestions.ipynb`
* `btw17_analysis.ipynb`