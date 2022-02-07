# Topics in Tweets and Query Suggestions (BTW 17)

This is the repository for my Bachelor-Thesis **Topic-Analyse politischer Tweets und Suchvorschläge zur Bundestagswahl 2017** in Data and Information Science at the University of Applied Sciences Cologne.
#### Structure of this repository
* `notebooks\`: .ipynb notebooks to work with, analyze and visualize data
* `reports\`: .html version of the notebooks
* `scripts\`: .py scripts for functions used in the notebooks

#### Notebook description
| filename                    | description                                                                                                                                                         |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `btw17_analysis.ipynb` | Visual analysis of the suggestions and hashtag matching. |
| `btw17_compare_topics_suggestions.ipynb` | Get similarity score for suggestions and hashtag topics. |
| `btw17_querysuggestions_data_structure.ipynb` | First analysis of the BTW17 query suggestions dataset. Simple matching of query suggestions. Plot used for Exposé. |
| `btw17_querysuggestions_dbscan.ipynb` | Retrieve relevant query suggestions (time filter) from BTW17 dataset. DBSCAN algorithm for clustering the query suggestions. |
| `btw17_twitter_lda.ipynb` | Latent Dirichlet Allocation on identified peaks per hashtag. |
| `btw17_twitter_lda_preprocess.ipynb` | Retrieve the peak date +- 3 days and the tweets per hashtag for LDA. |
| `btw17_twitter_wavelet.ipynb`   | Retrieve hashtag and their mentions per day for peak detection. Identify peaks per hashtag per Wavelet Transform and Kolmogorov-Zurbenko-Filter. |
| `plotting_wavelets.ipynb` | Plotting wavelets for thesis.

#### Recreating the Analysis
For recreating the results of this analysis you two datasets:
* ESuPol Query Suggestions Dataset from the University of Applied Sciences Köln
* Twitter Dataset from the University of Applied Sciences Lübeck, DOI: https://doi.org/10.3390/data2040034

With these two datasets, execute the described notebooks in the following order:
* `btw17_twitter_wavelet.ipynb`
* `btw17_twitter_lda_preprocess.ipynb`
* `btw17_twitter_lda.ipynb`
* `btw17_querysuggestions_dbscan.ipynb`
* `btw17_compare_topics_suggestions.ipynb`
* `btw17_analysis.ipynb`
