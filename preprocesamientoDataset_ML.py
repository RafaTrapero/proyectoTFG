import pandas as pd
from utils.utils import agregar_columnas_por_palabra, contar_palabras,calculate_non_zero_tfidf, cleanAndtokenize, filter_tfidf_by_threshold, remove_stopwords_from_column

df=pd.read_csv('twitter_covid_labelled_mickey.csv')

# Se eliminan aquellas filas que esten etiquetadas con 'U' (unverified), ya que no aportan información
df_filtrado = df[df['label'] != 'U']

# Número de filas sin 'U'
print('El número de filas despues de eliminar aquellas con label "U" es:', df_filtrado.shape[0])

# Se aplica la funcion de limpieza y se tokeniza cada tweet creando la columna 'tokenized_content'
df_filtrado['tokenized_content'] = df_filtrado['content'].apply(lambda x: cleanAndtokenize(x))

# Se eliminan las stopwords
df_filtrado=remove_stopwords_from_column(df_filtrado)

# Se eliminan aquellas columnas que no aportan información
df_filtrado = df_filtrado.drop(columns=["No.", "content", "source","sentiment"])

# Se crea un diccionario de las palabras presentes en la columna 'tokenized_content', dejando fuera aquellas con una ponderación de 0.
result = calculate_non_zero_tfidf(df_filtrado)  
# Se filtran aquellas palabras con una ponderacion TF-IDF entre 0.6 y 0.8
result=filter_tfidf_by_threshold(result,0.60,0.80)
# Finalmente se crea el diccionario
df_filtrado=agregar_columnas_por_palabra(df_filtrado,result)

# Se añade una columna adicional que indique el nº de palabras del tweet en cuestion 
df_filtrado['numWords'] = df_filtrado['tokenized_content'].apply(lambda x: contar_palabras(x))

df_filtrado=df_filtrado.drop(columns=['tokenized_content'])

df_filtrado['label'] = df_filtrado['label'].replace({'F': 0, 'T': 1})

# Se guarda como tsv el dataset final ya preprocesado
df_filtrado.to_csv('df_final.tsv', sep='\t', index=True)
