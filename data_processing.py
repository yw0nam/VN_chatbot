# %% 
import pandas as pd
import json
# %%
data = pd.read_csv('./data/QA.simple.tsv', sep='\t', names=['Q_person', 'Q', 'A_person', 'A'])
# %%
person = list(data['Q_person'].value_counts().index[data['Q_person'].value_counts() > 1200])
# %%
data['Q_person'] = data['Q_person'].map(lambda x: '<%s>'%x if x in person else '<某>')
data['A_person'] = data['A_person'].map(lambda x: '<%s>'%x if x in person else '<某>')
special_tokens = ['<%s>'%x for x in person] + ['<某>']
special_tokens = {'additional_special_tokens' :special_tokens}
# %%
with open('./data/special_token.json', 'w') as f:
    json.dump(special_tokens, f)
# %%
data['Q_text'] = data['Q_person']+data['Q']
data['A_text'] = data['A_person']+data['A']
data[['Q_text', 'A_text']].to_csv('./data/QA.tsv', sep='\t', index=False, header=False)
# %%
