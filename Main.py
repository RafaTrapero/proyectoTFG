#from machineLearning.NeuralNetwork import neural_network_tuning_cv_tensorflow
from utils.utils import agregar_columnas_por_palabra, contar_palabras, palabras_mas_frecuentes_por_label,agregar_columnas_por_palabra,palabras_por_label,columnas_a_eliminar,feature_selection_cascade,cleanAndtokenize,remove_stopwords_from_column,get_unique_words_by_label,calculate_non_zero_tfidf,filter_tfidf_by_threshold
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from machineLearning.logisticRegression import logistic_regression_hyperparameter_tuning_cv,logistic_regression_default,logistic_regression_tuning_cv
from machineLearning.decisionTree import decision_tree_tuning_cv,decision_tree_cv,decision_tree
from machineLearning.randomForest import randomForest,random_forest_hyperparameter_tuning_cv,random_forest_tuning_cv
#from machineLearning.NeuralNetwork import neural_network_tuning_cv,neural_network_tuning_cv_tensorflow
from sklearn.preprocessing import StandardScaler
# from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import DataLoader, TensorDataset
# from transformers import BertTokenizer, TFBertForSequenceClassification



#from machineLearning.NeuralNetwork import neuralNetwork

## PROCESAMIENTO DATASET   
df=pd.read_csv('twitter_covid_labelled_mickey.csv')

    # Se eliminan aquellas filas que esten etiquetadas con 'U' (unverified), ya que no aportan información
df = df[df['label'] != 'U']



# desde aqui hasta la linea 54 es analisis y ejemplos de lo que vamos a aplicar

## TEXT MINING

# Eliminamos aquellas filas que no estén etiquetadas con T o F
df_global = df[df['label'] != 'U']

# Se aplica la funcion de limpieza y tokenización a cada noticia (columna 'title')
df_global['tokenized_content'] = df_global['content'].apply(lambda x: cleanAndtokenize(x))
#df_global.to_csv('df_global.csv', index=True)


# Vemos las palabras más utilizadas por veracidad
# print("================== PALABRAS MAS USADAS POR VERACIDAD ==================")
# print(palabras_mas_frecuentes_por_label(df_global,10,'tokenized_content'))



# Podemos ver que los términos mas usados son articulos, preposiciones. Para ello eliminamos las stopwords.
df_global_sin_stopwords=remove_stopwords_from_column(df_global)
df_global_sin_stopwords.to_csv('df_global_sin_stopwords.csv', index=True)
# print(df_global_sin_stopwords)

print("================== PALABRAS MAS USADAS POR VERACIDAD (SIN STOPWORDS) ==================")
print(palabras_mas_frecuentes_por_label(df_global_sin_stopwords,20,'tokenized_content'))


print("=================================PALABRAS MAS USADAS POR VERACIDAD Y UNICAS POR LABEL====================================")
df_global_sin_stopwords['words_unique_label'] = get_unique_words_by_label(df_global_sin_stopwords)
# #df_global_sin_stopwords.to_csv('df_global_uniquewords.csv',index=True)

print(palabras_mas_frecuentes_por_label(df_global_sin_stopwords,20,'words_unique_label'))


result = calculate_non_zero_tfidf(df_global_sin_stopwords)  
result=filter_tfidf_by_threshold(result,0.60,0.80)

print("=================================PONDERACION TOKENS====================================")

for word, tfidf in result.items():
    print(f"Palabra: {word}, TF-IDF: {tfidf}")

# Inicializa un contador de palabras
word_count = 0

for word, tfidf in result.items():
    print(f"Palabra: {word}, TF-IDF: {tfidf}")
    # Incrementa el contador en 1 por cada palabra
    word_count += 1

# Imprime el número total de palabras
# print(f"Número total de palabras: {word_count}")

## PREPARAMOS EL DATASET PARA OBTENER EL FINAL
#df_global_sin_stopwords.to_csv('df_casi_final.csv',index=True)  
df_global_sin_stopwords = df_global_sin_stopwords.drop(columns=["No.", "content", "source"])
df_global_sin_stopwords.to_csv('def_sin_stopwords.tsv',sep='\t',index=True)

df_final=agregar_columnas_por_palabra(df_global_sin_stopwords,result)

df_final['numWords'] = df_final['tokenized_content'].apply(lambda x: contar_palabras(x))

#df_final.to_csv('df_final.tsv',sep='\t',index=True)
df_final=df_final.drop(columns=['tokenized_content'])
df_final=df_final.drop(columns=['words_unique_label'])
#df_final.to_csv('df_final.tsv',sep='\t',index=True)


## eliminio las filas que contengan los valores de !#VALUE y nan en sentiment
df_final['sentiment'] = df_final['sentiment'].replace({'#VALUE!': pd.NA, 'nan': pd.NA})

# Elimina las filas con NaN en la columna 'sentiment'

df_final = df_final.dropna(subset=['sentiment'])
df_final=df_final.drop(columns=['sentiment'])
columnasElimiar=columnas_a_eliminar(df_final,0.1) 
# print(columnasElimiar)
#df_final = df_final.drop(columns=columnasElimiar)
df_final.to_csv('df_final_main.tsv',sep='\t',index=True)
# veo manualmente que columnas no aportan info y las elimino
#df_final = df_final.drop(columns=['cato','joe','casama','tho','jmbrgnza','hbd','ogun','coronavirusinsa','tiger','jawan','hese','burkina','faso','cdo'])


# cambiamos los valores de F y T por 1 y 0
df_final['label'] = df_final['label'].replace({'F': 0, 'T': 1})
df_final.to_csv('df_final_main.tsv',sep='\t',index=True)
#df_final.to_csv('df_final.tsv',sep='\t',index=True)

num_columnas = df_final.shape[1]
# print("Número de columnas:", num_columnas)

## AQUI ACABA EL PREPROCESAMIENTO DE DATOS 
#############################################################################################################################################################################################
## COMIENZA LA PARTE DE MACHINE LEARNING

y = df_final['label'] # variable de estudio

X = df_final.drop('label', axis=1)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#df_final.to_csv('df_final.tsv',sep='\t',index=True)
#print('x train',X_train)
#print('x test',X_test)
#print('Y train',y_train)
#print('y test',y_test)
#X_train_selected, X_test_selected = feature_selection_cascade(X_train, y_train, X_test,40,0.2)
#df_final.to_csv('df_final_40_mejores_variables.tsv',sep='\t',index=True)
## 1) Logistic Regression
#print('------------------ MEJOR MODELO LOGISTICO--------------')
#mejor_modelo_logistico = logistic_regression_hyperparameter_tuning_cv(X_train, y_train, X_test, y_test, 200)

#print('--------------- DEFAULT----------------')
#lr_default=logistic_regression_default(X_train, y_train, X_test, y_test)
# print('--------------- TUNNING CV----------------')
#lr_tunning_cv=logistic_regression_hyperparameter_tuning_cv(X_train_selected, y_train, X_test_selected, y_test,50)
print('--------------- LR CV----------------')
lr_cv=logistic_regression_tuning_cv(X_train, y_train, X_test, y_test,20,50)


# # Contar la cantidad de ejemplos por clase
conteo_clases = df_final['label'].value_counts()

# # Calcular el porcentaje de cada clase
porcentaje_clases = conteo_clases / len(df_final) * 100

# # Imprimir los resultados
print("Porcentaje de cada clase:")
print(porcentaje_clases)

## 2) Decision Tree



# Entrenar y evaluar el modelo de árbol de decisión con las características seleccionadas
print('--------------DECISION TREE--------------')
#dt_model = decision_tree_tuning_cv(X_train, y_train, X_test, y_test,50)



## 3) Random Forest (la ejecucion dura para siempre)
print('----------RANDOM FOREST----------')
#rf_model=random_forest_tuning_cv(X_train, y_train, X_test, y_test,20)

print('--------------------NN---------------------')
#nn_model=neural_network_tuning_cv_tensorflow(X_train, y_train, X_test, y_test,20)

print('---------------------------DEEP LEARNING------------------------------')
# # Crear un modelo básico
# model = Sequential()
# model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# # Compilar el modelo
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Entrenar el modelo
# model.fit(X_train, y_train, epochs=10, batch_size=32)

# # Evaluar el modelo
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Accuracy: {accuracy}')

# Hacer predicciones
#predictions = model.predict(X_test)

###############################################





"""""""""""""""""""""""""""""""""""""""""""""


# PREPARAMOS LOS DATOS PARA PODER USARLOS EN LOS DISTINTOS ALGORITMOS


X=df_global2[['title','text','subject','date']]
Y=df_global2['veracity'] # declaramos la variable de estudio

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2),tokenizer=cleanAndtokenize,stop_words=stop_words) #min_df el numero minimo de documentos donde tienen que aparecer las palabras // ngran_range=el numero de gramas (mono gramas y bigramas)

X_text = tfidf.fit_transform(df_global2['title'])


print("Dimensiones de X:", X.shape)
print("Dimensiones de  Y: ", Y.shape)

print("Dimensiones de X_text:", X_text.shape)
print("Dimensiones de  Y: ", Y.shape)
# vemos que el numero de dimensiones de X_text no es el mismo que en Y. Por ello ejecutamos el siguiente codigo YA ME FUNCIONAAAAAAAAAAAAAAAAAAAAAAAAAAAA


# dividimos el conjunto en testing y training
X_train, X_test, y_train, y_test = train_test_split(X_text, Y, test_size=0.2, random_state=42)


# Seguimos con tf-idf
# tfidf_vectorizador.fit(X_train)

# tfidf_train = tfidf_vectorizador.transform(X_train)
# tfidf_test  = tfidf_vectorizador.transform(X_test)

# MACHINE LEARNING

# 1) Logistic Regression
#print("Precisión del modelo LR: ", logisticRegression(X_train,y_train,X_test,y_test))

# 2) Decision Tree
#print("Precision del modelo DT: ",decisionTree(X_train,y_train,X_test,y_test))

# 3) Random Forest
#print("Precision del modelo DT: ",randomForest(X_train,y_train,X_test,y_test))

# 4) Neural Network
#print(X_train.shape[1]) ##vemos el valor para pasarlo a input_dim 
print("Precision del modelo NN: ",neuralNetwork(X_train,y_train,X_test,y_test))
#print(type(y_train))


"""


 