from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')


def palabras_mas_frecuentes_por_label(df, n, column_name):
    palabras_frecuentes = []

    # Se obtienen los valores únicos de la columna 'label'
    labels_unicos = df['label'].unique()

    # Se iteran a través de los valores únicos en 'label'
    for label in labels_unicos:
        # Filtracio  del DataFrame para obtener solo las filas con el valor de 'label' actual
        subconjunto = df[df['label'] == label]

        # Se combinan todos los tokens en una lista
        tokens = [word for sublist in subconjunto[column_name] for word in sublist]

        conteo_palabras = Counter(tokens)
        
        palabras_mas_frecuentes = conteo_palabras.most_common(n)

        for palabra, frecuencia in palabras_mas_frecuentes:
            palabras_frecuentes.append((label, palabra, frecuencia))


    df_resultados = pd.DataFrame(palabras_frecuentes, columns=['Label', 'Palabra', 'Frecuencia'])
    
    # Visualización la tabla
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Creación de la tabla
    table = ax.table(cellText=df_resultados.values,
                     colLabels=df_resultados.columns,
                     cellLoc='center', loc='center',
                     colColours=['blue'] * len(df_resultados.columns))  

    # Estilo para el header
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight='bold', color='white')

    plt.show()

    return df_resultados


# en este caso, el valor de las columnas (tokens) sera 1 o 0 si ese token se encuentra en el tweet.
def agregar_columnas_por_palabra(df, result):

    for word in result.keys():
        # Se crea una nueva columna con el nombre de la palabra
        df[word] = 0 


        for index, row in df.iterrows():
            # Verifica si la palabra está en el array 'tokenized_content'
            if word in row['tokenized_content']:
                # Si la palabra está presente, asigna 1
                df.at[index, word] = 1

    return df


def contar_palabras(arr):
    return len(arr)

def crear_tabla_tf_idf(result, num_elementos):
    # Se ordena el diccionario de manera descendente y se seleccionan los primeros 'num_elementos'
    sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)[:num_elementos]

    # Se convierte el diccionario en tuplas ('Palabra', 'TF-IDF')
    data = [(word, tfidf) for word, tfidf in sorted_result]

    df = pd.DataFrame(data, columns=['Palabra', 'TF-IDF'])

    # Visualización de  la tabla
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Creación de la tabla
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colColours=[(0.8, 0.8, 1.0)] * len(df.columns))  # Establecer color de fondo para el encabezado (RGB = (0.8, 0.8, 1.0))

    # Estilo para el header
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight='bold', color='black')  # Cambiar el color del texto del encabezado a negro

    plt.show()

    return df

def cleanAndtokenize(texto):
    nuevo_texto = texto.lower()
    nuevo_texto = re.sub('http\S+', ' ', nuevo_texto)
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    nuevo_texto = re.sub(regex , ' ', nuevo_texto)
    
    # Eliminación de espacios en blanco
    nuevo_texto = re.sub("\\s+", ' ', nuevo_texto)
    
    # Tokenización por palabras individuales
    nuevo_texto = nuevo_texto.split(sep = ' ')
    
    # Eliminación de tokens con una longitud < 2
    nuevo_texto = [token for token in nuevo_texto if len(token) > 1]
    
    return(nuevo_texto)


# Función para eliminar stopwords de una lista de palabras
def remove_stopwords(word_list):
    stop_words = set(stopwords.words('english'))
    return [word for word in word_list if word not in stop_words]

# Función para aplicar la eliminación de stopwords a una columna en el DataFrame
def remove_stopwords_from_column(df):
    df['tokenized_content'] = df['tokenized_content'].apply(remove_stopwords)
    return df

def get_unique_words_by_label(df):
    unique_words_by_label = {}  

    for index, row in df.iterrows():
        label = row['label']
        tokenized_content = row['tokenized_content']

        if label not in unique_words_by_label:
            unique_words_by_label[label] = set(tokenized_content)
        else:
            unique_words_by_label[label] = unique_words_by_label[label].difference(tokenized_content)

    # lista de palabras únicas por etiqueta
    unique_words_list = []
    for index, row in df.iterrows():
        label = row['label']
        unique_words_list.append(list(unique_words_by_label[label]))

    return unique_words_list    


def calculate_non_zero_tfidf(df):
    corpus = df['tokenized_content'].apply(lambda x: ' '.join(x))  # Convierte la lista de palabras en texto


    tfidf_vectorizer = TfidfVectorizer()
    
    # Ajusta el vectorizador al corpus completo
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Obtención las características (palabras) y sus ponderaciones TF-IDF
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_values = tfidf_matrix.toarray()

    words_tfidf = {}
    for i, document in enumerate(corpus):
        for j, feature in enumerate(feature_names):
            tfidf = tfidf_values[i][j]
            if tfidf != 0.0:
                words_tfidf[feature] = tfidf

    return words_tfidf

def filter_tfidf_by_threshold(tfidf_dict, lower_threshold, upper_threshold):
    filtered_tfidf = {}

    for word, tfidf in tfidf_dict.items():
        if lower_threshold <= tfidf <= upper_threshold:
            filtered_tfidf[word] = tfidf

    return filtered_tfidf