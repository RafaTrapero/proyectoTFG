from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, auc, roc_curve
import pandas as pd

def decision_tree_tuning_cv(X_train, y_train, X_test, y_test, cv1, cv2):

    dt_classifier = DecisionTreeClassifier(random_state=42)

    # Parámetros para la hiperparametrización
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [1, 3, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt']  
    }
    
    # NO CV
    
    # Entrenamiento del modelo
    dt_classifier.fit(X_train, y_train)
    
    # Predicciones en el conjunto de prueba 
    y_pred_test_no_cv = dt_classifier.predict(X_test)
    
    # Evaluación del rendimiento
    accuracy_test_no_cv = accuracy_score(y_test, y_pred_test_no_cv)
    confusion_mat_test_no_cv = confusion_matrix(y_test, y_pred_test_no_cv)
    classification_rep_test_no_cv = classification_report(y_test, y_pred_test_no_cv)
    
    print(f'Accuracy en conjunto de prueba sin CV: {accuracy_test_no_cv:.4f}')
    print('\nConfusion Matrix en conjunto de prueba sin CV:')
    print(confusion_mat_test_no_cv)
    print('\nClassification Report en conjunto de prueba sin CV:')
    print(classification_rep_test_no_cv)
    
    # Obtención de la importancia de las características
    feature_importances = dt_classifier.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print('\nImportancia de las características:')
    print(feature_importance_df)
    
    # Curva ROC 
    plt.figure()
    lw = 2
    y_prob_test_no_cv = dt_classifier.predict_proba(X_test)[:, 1]
    fpr_no_cv, tpr_no_cv, _ = roc_curve(y_test, y_prob_test_no_cv)
    roc_auc_no_cv = auc(fpr_no_cv, tpr_no_cv)
    plt.plot(fpr_no_cv, tpr_no_cv, color='darkorange', lw=lw, label=f'ROC sin CV (AUC = %0.2f)' % roc_auc_no_cv)

    
    
    # PRIMER VALOR DE CV
    
    # Hiperparametrización con GridSearchCV 
    grid_search = GridSearchCV(dt_classifier, param_grid, cv=cv1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Obtención del mejor modelo (gracias a la hiperparametrización) 
    best_dt_model_cv1 = grid_search.best_estimator_

    """
    En este caso, el uso de X_test en el método .predict() tiene sentido, ya que estás evaluando el rendimiento del modelo seleccionado 
    (el mejor modelo obtenido después de la hiperparametrización) en un conjunto de datos independiente, es decir, en datos que el modelo 
    no ha visto durante el entrenamiento o la validación cruzada.
    """
    # Predicciones en conjunto de prueba
    y_pred_test_cv1 = best_dt_model_cv1.predict(X_test)

    # Evaluación del rendimiento
    accuracy_cv1 = accuracy_score(y_test, y_pred_test_cv1)
    confusion_mat_cv1 = confusion_matrix(y_test, y_pred_test_cv1)
    classification_rep_cv1 = classification_report(y_test, y_pred_test_cv1)
    
    print(f'\nAccuracy promedio en validación cruzada en conjunto de entrenamiento ({cv1}-fold): {accuracy_cv1:.4f}')
    print('\nConfusion Matrix en validación cruzada en conjunto de entrenamiento:')
    print(confusion_mat_cv1)
    print('\nClassification Report en validación cruzada en conjunto de entrenamiento:')
    print(classification_rep_cv1)
    
    # Imprimir los valores de los hiperparámetros del mejor modelo obtenido con cv1
    print("\nMejores hiperparámetros del modelo con CV1:")
    print(best_dt_model_cv1.get_params())
    
    # Curva ROC 
    y_prob_test_cv1 = best_dt_model_cv1.predict_proba(X_test)[:, 1]
    fpr_cv1, tpr_cv1, _ = roc_curve(y_test, y_prob_test_cv1)
    roc_auc_cv1 = auc(fpr_cv1, tpr_cv1)
    plt.plot(fpr_cv1, tpr_cv1, color='green', lw=lw, label=f'ROC con CV ({cv1}-fold) (AUC = %0.2f)' % roc_auc_cv1)

    
    # SEGUNDO VALOR DE CV

    # Hiperparametrización con GridSearchCV 
    grid_search = GridSearchCV(dt_classifier, param_grid, cv=cv2, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Obtención del mejor modelo (gracias a la hiperparametrización) 
    best_dt_model_cv2 = grid_search.best_estimator_

    # Predicciones del conjunto de prueva
    y_pred_test_cv2 = best_dt_model_cv2.predict(X_test)

    # Evaluación del rendimiento
    accuracy_cv2 = accuracy_score(y_test, y_pred_test_cv2)
    confusion_mat_cv2 = confusion_matrix(y_test, y_pred_test_cv2)
    classification_rep_cv2 = classification_report(y_test, y_pred_test_cv2)
    
    print(f'\nAccuracy promedio en validación cruzada en conjunto de entrenamiento ({cv2}-fold): {accuracy_cv2:.4f}')
    print('\nConfusion Matrix en validación cruzada en conjunto de entrenamiento:')
    print(confusion_mat_cv2)
    print('\nClassification Report en validación cruzada en conjunto de entrenamiento:')
    print(classification_rep_cv2)

    # Imprimir los valores de los hiperparámetros del mejor modelo obtenido con cv2
    print("\nMejores hiperparámetros del modelo con CV2:")
    print(best_dt_model_cv2.get_params())
    
    # Curva ROC 
    y_prob_test_cv2 = best_dt_model_cv2.predict_proba(X_test)[:, 1]
    fpr_cv2, tpr_cv2, _ = roc_curve(y_test, y_prob_test_cv2)
    roc_auc_cv2 = auc(fpr_cv2, tpr_cv2)
    plt.plot(fpr_cv2, tpr_cv2, color='blue', lw=lw, label=f'ROC con CV ({cv2}-fold) (AUC = %0.2f)' % roc_auc_cv2)

   
    # Se trazan las curvas ROC
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig('curva_roc_dt.jpg')
    plt.close()

    return dt_classifier
