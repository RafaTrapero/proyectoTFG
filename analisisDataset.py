import pandas as pd
from utils.utils import crear_tabla_tf_idf, grafica_barras, grafica_value_porcentaje, palabras_mas_frecuentes_por_label,calculate_non_zero_tfidf, cleanAndtokenize, filter_tfidf_by_threshold, get_unique_words_by_label, remove_stopwords_from_column

df=pd.read_csv('twitter_covid_labelled_mickey.csv')

# Se eliminan aquellas filas que esten etiquetadas con 'U' (unverified), ya que no aportan información
df_filtrado = df[df['label'] != 'U']

# Se aplica la funcion de limpieza y se tokeniza cada tweet creando la columna 'tokenized_content'
df_filtrado['tokenized_content'] = df_filtrado['content'].apply(lambda x: cleanAndtokenize(x))

print("================== PALABRAS MAS USADAS POR VERACIDAD ==================")
palabras_mas_frecuentes_por_label(df_filtrado,10,'tokenized_content')

# Se eliminan las stopwords
df_filtrado=remove_stopwords_from_column(df_filtrado)

print("================== PALABRAS MAS USADAS POR VERACIDAD (SIN STOPWORDS) ==================")
palabras_mas_frecuentes_por_label(df_filtrado,10,'tokenized_content')

print("=================================PALABRAS MAS USADAS POR VERACIDAD Y UNICAS POR LABEL====================================")
df_filtrado['words_unique_label'] = get_unique_words_by_label(df_filtrado)
palabras_mas_frecuentes_por_label(df_filtrado,20,'words_unique_label')



print("=================================PONDERACION TOKENS====================================")
result = calculate_non_zero_tfidf(df_filtrado)  
result=filter_tfidf_by_threshold(result,0.60,0.80)

crear_tabla_tf_idf(result,30)

# Gráfica de barras para observar la distribución de los tweets por label
grafica_barras('twitter_covid_labelled_mickey.csv')

# Distribución valores nulos en la columna 'sentiment'
grafica_value_porcentaje('twitter_covid_labelled_mickey.csv')