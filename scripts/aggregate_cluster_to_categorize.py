import pandas as pd

df = pd.read_json('../../data/BTW17_Suggestions/suggestions/cluster.json')
output = df[['cluster','suggestion']].groupby(['cluster'])['suggestion'].sum()
output.to_csv(('../../data/BTW17_Suggestions/suggestions/cluster_categories.csv'))