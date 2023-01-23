import pandas as pd
import glob
import re
from tqdm import tqdm

filenames = glob.glob('./extract/*.csv')
ids = []
texts = []
for filename in filenames:
    id = re.search('/([^/]*)\.csv', filename).group(1)
    ids.append(int(id))
ids = sorted(ids)

for id in tqdm(ids):
    df = pd.read_csv(f"./extract/{id}.csv")
    Texts = df['text']
    text = ''
    for Text in Texts:
        text += str(Text).lower() + ' [SEP] '
    text = re.sub('\.', '', text)
    text = re.sub('[^a-z0-9A-Z\[\]]', ' ', text)
    texts.append(text)

claims = []
labels = []

df = pd.read_json('./train.json')
for i in range(len(df['metadata'])):
    claim = df['metadata'][i]['claim']
    claim = claim.lower()
    claim = re.sub('\.', '', claim)
    claim = re.sub('[^a-z0-9]', ' ', claim)
    claims.append(claim)
    label = df['label'][i]['rating']
    labels.append(label)

end = len(df['metadata'])
output = {'id': ids[:end], 'text': texts[:end], 'claim': claims, 'label': labels}
DF = pd.DataFrame.from_dict(output)
DF.to_csv('train.csv', index=False)

claims = []
labels = []
df = pd.read_json('./valid.json')
for i in range(len(df['metadata'])):
    claim = df['metadata'][i]['claim']
    claim = claim.lower()
    claim = re.sub('\.', '', claim)
    claim = re.sub('[^a-z0-9]', ' ', claim)
    claims.append(claim)
    label = df['label'][i]['rating']
    labels.append(label)

end2 = len(df['metadata'])
output = {'id': ids[end:end+end2], 'text': texts[end:end+end2], 'claim': claims, 'label': labels}
DF = pd.DataFrame.from_dict(output)
DF.to_csv('valid.csv', index=False)

claims = []
df = pd.read_json('./test.json')
for i in range(len(df['metadata'])):
    claim = df['metadata'][i]['claim']
    claim = claim.lower()
    claim = re.sub('\.', '', claim)
    claim = re.sub('[^a-z0-9]', ' ', claim)
    claims.append(claim)

output = {'id': ids[end+end2:], 'text': texts[end+end2:], 'claim': claims}
DF = pd.DataFrame.from_dict(output)
DF.to_csv('test.csv', index=False)
