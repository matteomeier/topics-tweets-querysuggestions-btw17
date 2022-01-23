import pandas as pd

df = pd.read_json('../../data/BTW17_Suggestions/suggestions/cluster.json')

df['sugg'] = df['suggestion'].apply(lambda x: ' '.join([sugg for sugg in x]))

output = {'cluster':[],'sugg':[]}

for cluster in df['cluster'].unique():
    output['cluster'].append(cluster)
    tmp = df[df['cluster']==cluster]
    output['sugg'].append(tmp['sugg'].str.cat(sep=', '))

pd.DataFrame(data=output).to_csv(('../../data/BTW17_Suggestions/suggestions/cluster_categories.csv'))