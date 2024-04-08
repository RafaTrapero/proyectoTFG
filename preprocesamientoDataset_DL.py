import pandas as pd

# Cargar y preprocesar datos
df = pd.read_csv('twitter_covid_labelled_mickey.csv')
df = df.drop(columns=['No.', 'source', 'sentiment', 'reply numbers', 'retweet numbers', 'likes numbers'])
df = df[df['label'] != 'U']

# df=pd.concat([df_true_downsampled,df_false])
df['label'] = df['label'].replace({'F': 0, 'T': 1})

df.to_csv('df_final_DL.csv', index=True)