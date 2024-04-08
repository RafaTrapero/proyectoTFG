from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict,StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def logistic_regression_tuning_cv(X_train, y_train, X_test, y_test, cv1, cv2):

    model = LogisticRegression(random_state=42, C=0.1, class_weight='balanced')
    
    # SIN CV
    
    # Entrenamiento del modelo
    model.fit(X_train, y_train)

    # Predicciones en el conjunto de prueba
    y_pred_test_no_cv = model.predict(X_test)
    
    # Evaluación del rendimiento del modelo sin CV
    accuracy_test_no_cv = accuracy_score(y_test, y_pred_test_no_cv)
    confusion_mat_test_no_cv = confusion_matrix(y_test, y_pred_test_no_cv)
    classification_rep_test_no_cv = classification_report(y_test, y_pred_test_no_cv)
    
    print(f'Accuracy en conjunto de prueba sin CV: {accuracy_test_no_cv:.4f}')
    print('\nConfusion Matrix en conjunto de prueba sin CV:')
    print(confusion_mat_test_no_cv)
    print('\nClassification Report en conjunto de prueba sin CV:')
    print(classification_rep_test_no_cv)

    # Curva ROC
    plt.figure()
    lw = 2
    y_prob_test_no_cv = model.predict_proba(X_test)[:, 1]
    fpr_no_cv, tpr_no_cv, _ = roc_curve(y_test, y_prob_test_no_cv)
    roc_auc_no_cv = auc(fpr_no_cv, tpr_no_cv)
    plt.plot(fpr_no_cv, tpr_no_cv, color='darkorange', lw=lw, label='ROC sin CV (AUC = %0.2f)' % roc_auc_no_cv)
    
    
    # PRIMER VALOR DE CV

    # Predicciones del conjunto de prueba(en este caso el conjunto de prueva es x_train por lo que explico abajo)
    """"
    Si se usaran X_test e y_test en lugar de X_train e y_train, se estaría evaluando el modelo en un conjunto de datos 
    independiente, lo cual es útil para evaluar el rendimiento final del modelo, pero no proporcionaría una estimación 
    adecuada del rendimiento durante el entrenamiento o validación del modelo.
    Al usar `X_train` e `y_train` en la función `cross_val_predict`, se está evaluando el rendimiento del modelo en datos de entrenamiento 
    que no se han utilizado durante el entrenamiento de cada modelo individual en cada fold. Esto proporciona una estimación más realista 
    del rendimiento del modelo en datos no vistos.  
    """
    y_pred__cv1 = cross_val_predict(model, X_train, y_train, cv=StratifiedKFold(n_splits=cv1, shuffle=True, random_state=42))
    
    # Evaluación del rendimiento 
    accuracy_cv1 = accuracy_score(y_train, y_pred__cv1)
    confusion_mat_cv1 = confusion_matrix(y_train, y_pred__cv1)
    classification_rep_cv1 = classification_report(y_train, y_pred__cv1)

    print(f'\nAccuracy promedio en validación cruzada en conjunto de entrenamiento ({cv1}-fold): {accuracy_cv1:.4f}')
    print('\nConfusion Matrix en validación cruzada en conjunto de entrenamiento:')
    print(confusion_mat_cv1)
    print('\nClassification Report en validación cruzada en conjunto de entrenamiento:')

    # Curva ROC
    y_prob_test_cv1 = cross_val_predict(model, X_test, y_test, cv=StratifiedKFold(n_splits=cv1, shuffle=True, random_state=42), method='predict_proba')[:, 1]
    fpr_cv1, tpr_cv1, _ = roc_curve(y_test, y_prob_test_cv1)
    roc_auc_cv1 = auc(fpr_cv1, tpr_cv1)
    plt.plot(fpr_cv1, tpr_cv1, color='green', lw=lw, label=f'ROC con CV ({cv1}-fold) (AUC = %0.2f)' % roc_auc_cv1)
    
    
    print(classification_rep_cv1)
    
    # SEGUNDO VALOR DE CV

    # Predicciones
    y_pred_cv2 = cross_val_predict(model, X_train, y_train, cv=StratifiedKFold(n_splits=cv2, shuffle=True, random_state=42))
    
    # Evaluación del rendimiento 
    accuracy_cv2 = accuracy_score(y_train, y_pred_cv2)
    confusion_mat_cv2 = confusion_matrix(y_train, y_pred_cv2)
    classification_rep_cv2 = classification_report(y_train, y_pred_cv2)

    print(f'\nAccuracy promedio en validación cruzada en conjunto de entrenamiento ({cv2}-fold): {accuracy_cv2:.4f}')
    print('\nConfusion Matrix en validación cruzada en conjunto de entrenamiento:')
    print(confusion_mat_cv2)
    print('\nClassification Report en validación cruzada en conjunto de entrenamiento:')
    print(classification_rep_cv2)

    # Curva ROC
    y_prob_test_cv2 = cross_val_predict(model, X_test, y_test, cv=StratifiedKFold(n_splits=cv2, shuffle=True, random_state=42), method='predict_proba')[:, 1]
    fpr_cv2, tpr_cv2, _ = roc_curve(y_test, y_prob_test_cv2)
    roc_auc_cv2 = auc(fpr_cv2, tpr_cv2)

    # Las curvas ROC son trazadas
    plt.plot(fpr_cv2, tpr_cv2, color='blue', lw=lw, label=f'ROC con CV ({cv2}-fold) (AUC = %0.2f)' % roc_auc_cv2)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    
    plt.savefig('curva_roc_lr.jpg')
    plt.close()

    return model
