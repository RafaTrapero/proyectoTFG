import pandas as pd
from sklearn.model_selection import train_test_split
from logisticRegression import logistic_regression_tuning_cv
from decisionTree import decision_tree_tuning_cv
from randomForest import random_forest_tuning_cv
from neuralNetwork import neural_network_tuning_cv


df=pd.read_csv('df_final.tsv', sep='\t')

y = df['label'] # variable de estudio

X = df.drop('label', axis=1)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# REGRESIÓN LOGÍSTICA
lr_cv=logistic_regression_tuning_cv(X_train, y_train, X_test, y_test,20,50)

# DECISION TREE
dt_model = decision_tree_tuning_cv(X_train, y_train, X_test, y_test,20,50)

# RANDOM FOREST
rf_model=random_forest_tuning_cv(X_train, y_train, X_test, y_test,20,50)

# NEURAL NETWORK
nn_model=neural_network_tuning_cv(X_train, y_train, X_test, y_test,20,50)